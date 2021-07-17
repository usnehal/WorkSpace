import tensorflow as tf
import re
import pandas as pd
import pandas as pd
import os
import pickle5 as pickle
import socket
import numpy as np
import tensorflow as tf
from    sklearn.metrics import accuracy_score
import  socket

from common.logger import Logger
from common.config import Config
from common.constants import bcolors, test

cfg = Config()

# def read_image(image_path,label):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, (299, 299))
#     image = tf.cast(image, tf.float32)
#     image /= 127.5
#     image -= 1.
#     return image, label

def read_image(image_path, height=299, width=299):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (height, width))
    image = tf.cast(image, tf.float32)
    image /= 127.5
    image -= 1.
    return image

def filt_text(text):
    filt=['<start>','<unk>','<end>'] 
    temp= text.split()
    [temp.remove(j) for k in filt for j in temp if k==j]
    text=' '.join(temp)
    return text

    # define a function to clean text data
def extract_jpg_caption(images_path, line):
    char_filter = r"[^\w]"

    jpg_path = None
    caption = None

    jpg_position = line.find(".jpg")
    if(jpg_position != -1):
        jpg_path = images_path + '/' + line[:jpg_position+4]

        caption = line[jpg_position+5:].strip()

        # convert words to lower case
        caption = caption.lower()

        # split into words
        words = caption.split()

        # strip whitespace from all words
        words = [word.strip() for word in words]

        # join back words to get document
        caption = " ".join(words)

        # remove unwanted characters
        caption = re.sub(char_filter, " ", caption)

        # remove unwanted characters
        caption = re.sub(r"\.", " ", caption)

        # replace multiple whitespaces with single whitespace
        caption = re.sub(r"\s+", " ", caption)

        # strip whitespace from document
        caption = caption.strip()

        caption = '<start> ' + caption + ' <end>'

    return jpg_path, caption




class ImagesInfo:
    def __init__(self, cfg):
        self.cfg = cfg
        all_imgs = [cfg.images_path + line.rstrip() for line in open(self.cfg.list_file)]
        all_imgs = sorted(all_imgs)
        total_num_images = len(all_imgs)
        Logger.debug_print("The total images present in the dataset: {}".format(total_num_images))

        #Visualise both the images & text present in the dataset
        Logger.debug_print("The total images present in the dataset: {}".format(total_num_images))
        num_lines = sum(1 for line in open(self.cfg.text_file))
        Logger.debug_print("The total number of lines in the caption file: {}".format(num_lines))

        #store all the image id here
        self.all_img_id= [] 
        #store all the image path here
        self.all_img_vector=[]
        #store all the captions here
        self.annotations_dict = {} 
        self.annotations= [] 
        # list of all captions in word list format
        self.annotations_word_list = []

        self.df_captions = pd.read_csv(self.cfg.df_captions_csv)
        # self.df_instances = pd.read_csv(self.cfg.df_instances_csv)
        self.df_categories = pd.read_csv(self.cfg.df_categories_csv)

        with open(cfg.saved_model_path + '/classes.pickle', 'rb') as handle:
            self.classes = pickle.load(handle)

        doc = self.load_doc(self.cfg.text_file)

    def get_segmentation_id_indexes(self,image_file):
        image_file = os.path.basename(image_file)
        id_list = self.df_instances[self.df_instances.FileName == image_file].Caption.tolist()
        id_list = list(set(id_list))
        index_list = []
        for l in id_list:
            lst = self.df_categories.index[self.df_categories.id == l]
            index_list.append(lst[0])
                
        return index_list

    def get_segmentation_texts(self,id_list):
        text_list = []
        for l in id_list:
            text_list.append(self.classes[l])
        return text_list

    def load_doc(self, filename):
        #your code here
        file  = open(filename, 'r') 
        Lines = file.readlines() 
        
        text = ""
        count = 0
        for line in Lines:
            jpg_path, caption = extract_jpg_caption(self.cfg.images_path, line)
            if(jpg_path != None):
                self.all_img_id.append(count)
                self.all_img_vector.append(jpg_path)
                self.annotations.append(caption)

                caption_list = []
                if jpg_path in self.annotations_dict.keys():
                    caption_list = self.annotations_dict[jpg_path]
                caption_list.append(caption)
                self.annotations_dict[jpg_path] = caption_list

                word_list = caption.split()
                self.annotations_word_list.append(word_list)
                text += " " + caption
                count += 1
        file.close()
        return text

    def getImagePath(self, index):
        return self.all_img_vector[index]

    def getCaption(self, index):
        return self.annotations[index]        

    def getAllCaptions(self, img_path):
        real_appn = []
        real_caption_list = self.annotations_dict[img_path]
        for real_caption in real_caption_list:
            real_caption=filt_text(real_caption)
            real_appn.append(real_caption.split())
        return real_appn

def get_predictions(cfg, prediction_tensor):
    n = tf.squeeze(prediction_tensor).numpy()
    top_predictions = []
    top_predictions_prob = []
    index = 0
    for x in n:
        if x > cfg.PREDICTIONS_THRESHOLD:
            top_predictions.append(index)
            top_predictions_prob.append(int(x*100))
        index += 1
    return top_predictions, top_predictions_prob

def process_predictions(cfg, imagesInfo, ground_truth, top_predictions, top_predictions_prob):
    df = pd.DataFrame(columns=['id_index','probability'])
    predictions_str = ''
    predictions_length = len(top_predictions)
    assert(predictions_length == len(top_predictions_prob))
    for i in range(predictions_length):
        x = top_predictions_prob[i]
        p = top_predictions[i]
        if x > cfg.PREDICTIONS_THRESHOLD:
            top_predictions.append(p)
            predictions_str += "%s(%.2f) " % (imagesInfo.classes[p],x)
            df = df.append({'id_index':int(p), 'probability':x},ignore_index = True)

    df = df.sort_values('probability', ascending=False)
    sorted_predictions = df['id_index'].tolist()
    sorted_predictions = [int(x) for x in sorted_predictions]

    ground_truth_length = len(ground_truth)
    predictions_length = len(sorted_predictions)

    aligned_predictions = [-1] * ground_truth_length
    TP = 0
    for i in range(ground_truth_length):
        if(ground_truth[i] in sorted_predictions):
            aligned_predictions[i] = ground_truth[i]
            TP += 1
    accuracy = accuracy_score(ground_truth, aligned_predictions)

    top_1_accuracy = 0.0
    top_5_accuracy = 0.0
    precision = 0
    recall = 0
    if(predictions_length > 0):
        if(sorted_predictions[0] in ground_truth):
            top_1_accuracy = 1.0
        for i in range(5):
            if((i < predictions_length) and (sorted_predictions[i] in ground_truth)):
                top_5_accuracy = 1.0

        precision = TP / predictions_length
    if(ground_truth_length > 0):
        recall = TP / ground_truth_length
    return accuracy, top_1_accuracy,top_5_accuracy,precision,recall, top_predictions, predictions_str

# extractor = tf.keras.Model(inputs=model.inputs,
#                         outputs=[layer.output for layer in model.layers])
# features = extractor(sample_img_batch)
# for i, feature in enumerate(features):
#     print(i, tf.shape(feature))

# import  matplotlib.pyplot as plt
# image = tf.squeeze(sample_img_batch,[0])
# plt.imshow(image)