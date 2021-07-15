#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import all the required libraries
import  time
import  pandas as pd
import  numpy as np
# from    skimage import io
# import  random
# from    collections import Counter
# from    tensorflow.keras.preprocessing.text import Tokenizer
import  tensorflow as tf
# from    tensorflow import keras
# from    tensorflow.keras import layers,Model
from    tqdm import tqdm
# from    nltk.translate.bleu_score import sentence_bleu
# import  socket
# import  pickle5 as pickle
# from    tensorflow.keras.activations import tanh
# from    tensorflow.keras.activations import softmax
# import  matplotlib.pyplot as plt
import  time
import  argparse
from    sklearn.metrics import accuracy_score
import  tensorflow_datasets as tfds
import  functools


# In[2]:


from Helper import Config, ImagesInfo, Logger, Client, TimeKeeper
from Helper import read_image


# In[3]:


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server', action='store', type=str, required=False)
parser.add_argument('-t', '--test_number', action='store', type=int, required=False)
parser.add_argument('-v', '--verbose', action='store', type=int, required=False)
parser.add_argument('-i', '--image_size', action='store', type=int, required=False)
parser.add_argument('-m', '--max_tests', action='store', type=int, required=False)
args, unknown = parser.parse_known_args()
print(args.server)

server_ip = args.server
test_number = args.test_number
verbose = args.verbose
image_size = args.image_size
max_tests = args.max_tests

if(verbose == None):
    verbose = 1

if(test_number == None):
    test_number = 3

if(max_tests == None):
    max_tests = 100

test_scenarios = {  1:"Complete jpg file buffer transfer", 
                    2:"Decoded image buffer transfer",
                    3:"Decoded image buffer transfer with zlib compression"}

if(image_size == None):
    image_size = 299


# In[ ]:


Logger.set_log_level(verbose)
tk = TimeKeeper()
cfg = Config(server_ip)
client = Client(cfg)
imagesInfo = ImagesInfo(cfg)


# In[4]:


total_test_images = 100
batch_size = 32
PREDICTIONS_THRESHOLD = 0.4
h_image_height = image_size
h_image_width = image_size

Logger.milestone_print("Test scenario = %d %s" % (test_number, test_scenarios[test_number]))
Logger.milestone_print("Image shape = (%d %d)" % (h_image_height, h_image_width))
Logger.milestone_print("Max tests = %d" % (max_tests))


# In[6]:


# new_head_model = tf.keras.models.load_model(cfg.temp_path + '/new_head_model')
# new_tail_model = tf.keras.models.load_model(cfg.temp_path + '/new_tail_model')

new_head_model = tf.keras.models.load_model(cfg.saved_model_path + '/head_model')
new_tail_model = tf.keras.models.load_model(cfg.saved_model_path + '/tail_model')


# In[7]:


def process_predictions(ground_truth, prediction_tensor):
    n = tf.squeeze(prediction_tensor).numpy()
    df = pd.DataFrame(columns=['id_index','probability'])
    predictions_str = ''
    top_predictions = []
    index = 0
    for x in n:
        if x > PREDICTIONS_THRESHOLD:
            top_predictions.append(index)
            predictions_str += "%s(%.2f) " % (imagesInfo.classes[index],x)
            df = df.append({'id_index':int(index), 'probability':x},ignore_index = True)
        index += 1

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
    if(predictions_length > 0):
        recall = TP / ground_truth_length
    return accuracy, top_1_accuracy,top_5_accuracy,precision,recall, top_predictions, predictions_str


# In[8]:


def evaluate_classification(image):
    temp_input = tf.expand_dims(read_image(image), 0) 
    h = new_head_model(temp_input)
    s = new_tail_model(h)
    return s


# In[10]:


data_dir='/home/suphale/coco'
N_LABELS = 80
split_val = "validation[:20%]"
# h_image_height = 299
# h_image_width = 299


# In[11]:


class BoxField:
    BOXES = 'bbox'
    KEYPOINTS = 'keypoints'
    LABELS = 'label'
    MASKS = 'masks'
    NUM_BOXES = 'num_boxes'
    SCORES = 'scores'
    WEIGHTS = 'weights'

class DatasetField:
    IMAGES = 'images'
    IMAGES_INFO = 'images_information'
    IMAGES_PMASK = 'images_padding_mask'

def my_preprocess(inputs):
    image = inputs['image']
    image = tf.image.resize(image, (h_image_height, h_image_width))
    image = tf.cast(image, tf.float32)
    image /= 127.5
    image -= 1.

    targets = inputs['objects']

    image_information = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

    inputs = {DatasetField.IMAGES: image, DatasetField.IMAGES_INFO: image_information}

    # ground_truths = {
    #     BoxField.BOXES: targets[BoxField.BOXES] * tf.tile(image_information[tf.newaxis], [1, 2]),
    #     BoxField.LABELS: tf.cast(targets[BoxField.LABELS], tf.int32),
    #     BoxField.NUM_BOXES: tf.shape(targets[BoxField.LABELS]),
    #     BoxField.WEIGHTS: tf.fill(tf.shape(targets[BoxField.LABELS]), 1.0)
    # }
    ground_truths = tf.cast(targets[BoxField.LABELS], tf.int32)
    # ground_truths = tf.one_hot(ground_truths, depth=N_LABELS, dtype=tf.int32)
    # ground_truths = tf.reduce_sum(ground_truths, 0)
    # ground_truths = tf.greater( ground_truths, tf.constant( 0 ) )    
    # ground_truths = tf.where (ground_truths, 1, 0) 
    return image, ground_truths

def expand_dims_for_single_batch(image, ground_truths):
    image = tf.expand_dims(image, axis=0)
    ground_truths = tf.expand_dims(ground_truths, axis=0)
    return image, ground_truths


# In[12]:


ds_val = tfds.load(name="coco/2017", split=split_val, data_dir=data_dir, shuffle_files=False, download=False)
ds_val = ds_val.map(functools.partial(my_preprocess), num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_val = ds_val.map(expand_dims_for_single_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_val = ds_val.range(100)

# iterator = ds_val.make_one_shot_iterator()

ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)


# In[13]:


df = pd.DataFrame(columns=['img_path','ground_truth', 'top_predict', 'Prediction', 'accuracy', 'top_1_accuracy', 'top_5_accuracy', 'precision', 'recall', 'time'])

total_time = 0.0
# for test_index in range(10):
count = 0
max_test_images = max_tests
# for sample_img_batch, ground_truth in ds_val:
for i in tqdm(range(max_test_images)):
# for sample_img_batch, ground_truth in ds_val:
    count += 1
    sample_img_batch, ground_truth = next(iter(ds_val))
    tensor_shape = len(ground_truth.get_shape().as_list())
    if(tensor_shape > 1):
        ground_truth = tf.squeeze(ground_truth,[0])
    # print(ground_truth)
    t0= time.perf_counter()
    h = new_head_model(sample_img_batch)
    s = new_tail_model(h)
    t1 = time.perf_counter() - t0
    total_time = total_time + t1

    accuracy, top_1_accuracy,top_5_accuracy,precision,recall, top_predictions, predictions_str = process_predictions(ground_truth,s)
    df = df.append(
        {'image':i, 
        'ground_truth':(str(imagesInfo.get_segmentation_texts(ground_truth))),
        'top_predict':str(top_predictions),
        'Prediction':predictions_str,
        'accuracy':accuracy,
        'top_1_accuracy':top_1_accuracy,
        'top_5_accuracy':top_5_accuracy,
        'precision':precision,
        'recall':recall,
        'time':t1,
        },
        ignore_index = True)
    truth_str = ' '.join([str(elem) for elem in imagesInfo.get_segmentation_texts(ground_truth)])
    Logger.debug_print("ground_truth  : %s" % (truth_str))
    Logger.debug_print("Prediction    : %s" % (predictions_str))

df.to_csv(cfg.temp_path + '/results_'+cfg.timestr+'.csv')
av_column = df.mean(axis=0)

print(count)

Logger.milestone_print("accuracy        : %.2f" % (av_column.accuracy))
Logger.milestone_print("top_1_accuracy  : %.2f" % (av_column.top_1_accuracy))
Logger.milestone_print("top_5_accuracy  : %.2f" % (av_column.top_5_accuracy))
Logger.milestone_print("precision       : %.2f" % (av_column.precision))
Logger.milestone_print("recall          : %.2f" % (av_column.recall))
Logger.milestone_print("time            : %.2f" % (av_column.time))


# In[ ]:




