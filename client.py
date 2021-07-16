#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3


# In[2]:


import socket
import os
import json
import  time
import tensorflow as tf
import numpy as np
from   nltk.translate.bleu_score import sentence_bleu
import random
import  re
import sys
import argparse
import zlib
import pickle5 as pickle
import pandas as pd
from tqdm import tqdm
import  tensorflow_datasets as tfds
import  functools

from Helper import Config, ImagesInfo, Logger, Client, TimeKeeper
from Helper import read_image, filt_text, test, get_predictions, process_predictions


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

test_number = 0
if(test_number == None):
    test_number = test.STANDALONE
if(test_number == 0):
    test_number = test.STANDALONE
if(test_number == 1):
    test_number = test.JPEG_TRANSFER
if(test_number == 2):
    test_number = test.DECODED_IMAGE_TRANSFER
if(test_number == 3):
    test_number = test.DECODED_IMAGE_TRANSFER_ZLIB
if(test_number == 4):
    test_number = test.SPLIT_LAYER_3
if(test_number == 5):
    test_number = test.SPLIT_LAYER_3_ZLIB

test_scenarios = {  
        test.STANDALONE:                    "standalone processing at client device", 
        test.JPEG_TRANSFER:                 "Complete jpg file buffer transfer", 
        test.DECODED_IMAGE_TRANSFER:        "Decoded image buffer transfer",
        test.DECODED_IMAGE_TRANSFER_ZLIB:   "Decoded image buffer transfer with zlib compression",
        test.SPLIT_LAYER_3:                 "split model at layer 3",
        test.SPLIT_LAYER_3_ZLIB:            "split model at layer 3 with zlib compression",
        }

if(image_size == None):
    image_size = 299

if(max_tests == None):
    max_tests = 100
elif (((max_tests % 50) == 0) and (max_tests <= 5000)):
    max_tests = max_tests
else:
    print("max_tests must be multiple of 50 and less than or equal to 5000")
    exit(1)

print("Test scenario = %d %s" % (test_number, test_scenarios[test_number]))


# In[4]:


# tf.compat.v1.disable_eager_execution()


# In[5]:


Logger.set_log_level(verbose)
tk = TimeKeeper()
cfg = Config(server_ip)
client = Client(cfg)
imagesInfo = ImagesInfo(cfg)


# In[6]:


data_dir='/home/suphale/coco'
N_LABELS = 80
# split_val = "validation"
split_val = "validation[:20%]"
# split_val = "validation[:1%]"
# h_image_height = 299
# h_image_width = 299

h_image_height = image_size
h_image_width = image_size

Logger.event_print("Test scenario   : %d %s" % (test_number, test_scenarios[test_number]))
Logger.event_print("Image shape     : (%d %d)" % (h_image_height, h_image_width))
Logger.event_print("Max tests       : %d" % (max_tests))


# In[7]:


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
    img_path = inputs['image/filename']

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
    return image, ground_truths, img_path

def expand_dims_for_single_batch(image, ground_truths, img_path):
    image = tf.expand_dims(image, axis=0)
    ground_truths = tf.expand_dims(ground_truths, axis=0)
    return image, ground_truths, img_path


# In[8]:


def evaluate_file_over_server(file_name):
    with open(file_name, 'rb') as file_t:
        byte_buffer_to_send = bytearray(file_t.read())
        send_json_dict = {}
        send_json_dict['data_type'] = 'file'
        send_json_dict['file_name'] = file_name
        send_json_dict['data_size'] = (len(byte_buffer_to_send))
        send_json_dict['data_shape'] = "(%d,)" % (len(byte_buffer_to_send))
        # send_json_dict['data_buffer'] = blob_data

        app_json = json.dumps(send_json_dict)

        tk.logInfo(img_path, tk.I_BUFFER_SIZE, len(byte_buffer_to_send))

        tk.logTime(img_path, tk.E_START_COMMUNICATION)

        response = client.send_data(str(app_json), byte_buffer_to_send)

        tk.logTime(img_path, tk.E_STOP_COMMUNICATION)

        response = json.loads(response)

        predictions = response['predictions']
        predictions_prob = response['predictions_prob']
        # predictions = pickle.loads(predictions)
        tail_model_time = response['tail_model_time']
        tk.logInfo(img_path, tk.I_TAIL_MODEL_TIME, tail_model_time)

        return predictions, predictions_prob


# In[9]:


# tf.compat.v1.disable_eager_execution()


# In[10]:


if(test_number in [test.STANDALONE]):
    model = tf.keras.models.load_model(cfg.saved_model_path + '/model')
if(test_number in [test.JPEG_TRANSFER, test.DECODED_IMAGE_TRANSFER, test.DECODED_IMAGE_TRANSFER_ZLIB]):
    # head_model = tf.keras.models.load_model(cfg.saved_model_path + '/model')
    send_json_dict = {}
    send_json_dict['data_type'] = 'load_model_request'
    send_json_dict['model'] = '/model'
    app_json = json.dumps(send_json_dict)
    response = client.send_load_model_request(str(app_json))
    assert(response == 'OK')
if(test_number in [test.SPLIT_LAYER_3, test.SPLIT_LAYER_3_ZLIB]):
    head_model = tf.keras.models.load_model(cfg.saved_model_path + '/head_model')
    send_json_dict = {}
    send_json_dict['data_type'] = 'load_model_request'
    send_json_dict['model'] = '/tail_model'
    app_json = json.dumps(send_json_dict)
    response = client.send_load_model_request(str(app_json))
    assert(response == 'OK')


# In[11]:


def evaluate_standalone(sample_img_batch, img_path):
    # print(ground_truth)
    result = model(sample_img_batch)

    tk.logInfo(img_path, tk.I_BUFFER_SIZE, 0)

    tk.logTime(img_path, tk.E_START_COMMUNICATION)

    tk.logTime(img_path, tk.E_STOP_COMMUNICATION)

    predictions, predictions_prob = get_predictions(cfg, result)

    tk.logInfo(img_path, tk.I_TAIL_MODEL_TIME, 0)

    return predictions, predictions_prob


# In[12]:


def evaluate_over_server(image_tensor,file_name, zlib_compression=False):
    # image_tensor = read_image(file_name)
    # image_tensor = tf.expand_dims(image_tensor, 0) 

    image_np_array = image_tensor.numpy()

    byte_buffer_to_send = image_np_array.tobytes()
    if(zlib_compression == True):
        byte_buffer_to_send = zlib.compress(byte_buffer_to_send)

    type(byte_buffer_to_send)

    send_json_dict = {}
    send_json_dict['data_type'] = 'data'
    send_json_dict['file_name'] = file_name
    send_json_dict['data_size'] = (len(byte_buffer_to_send))
    send_json_dict['data_shape'] = image_np_array.shape
    if(zlib_compression == True):
        send_json_dict['zlib_compression'] = 'yes'
    else:
        send_json_dict['zlib_compression'] = 'no'

    app_json = json.dumps(send_json_dict)

    tk.logInfo(img_path, tk.I_BUFFER_SIZE, len(byte_buffer_to_send))

    tk.logTime(img_path, tk.E_START_COMMUNICATION)

    response = client.send_data(str(app_json), byte_buffer_to_send)

    tk.logTime(img_path, tk.E_STOP_COMMUNICATION)

    response = json.loads(response)

    predictions = response['predictions']
    predictions_prob = response['predictions_prob']
    tail_model_time = response['tail_model_time']
    tk.logInfo(img_path, tk.I_TAIL_MODEL_TIME, tail_model_time)

    return predictions, predictions_prob


# In[ ]:





# In[13]:


def evaluate_over_server_head_model(image_tensor,file_name, zlib_compression=False):

    # temp_input = tf.expand_dims(read_image(file_name), 0) 
    intermediate_tensor = head_model(image_tensor)
    image_np_array = intermediate_tensor.numpy()

    byte_buffer_to_send = image_np_array.tobytes()
    if(zlib_compression == True):
        byte_buffer_to_send = zlib.compress(byte_buffer_to_send)

    type(byte_buffer_to_send)

    send_json_dict = {}
    send_json_dict['data_type'] = 'data'
    send_json_dict['file_name'] = file_name
    send_json_dict['data_size'] = (len(byte_buffer_to_send))
    send_json_dict['data_shape'] = image_np_array.shape
    if(zlib_compression == True):
        send_json_dict['zlib_compression'] = 'yes'
    else:
        send_json_dict['zlib_compression'] = 'no'

    app_json = json.dumps(send_json_dict)

    tk.logInfo(img_path, tk.I_BUFFER_SIZE, len(byte_buffer_to_send))

    tk.logTime(img_path, tk.E_START_COMMUNICATION)

    response = client.send_data(str(app_json), byte_buffer_to_send)

    tk.logTime(img_path, tk.E_STOP_COMMUNICATION)

    response = json.loads(response)

    predictions = response['predictions']
    predictions_prob = response['predictions_prob']
    tail_model_time = response['tail_model_time']
    tk.logInfo(img_path, tk.I_TAIL_MODEL_TIME, tail_model_time)

    return predictions, predictions_prob


# In[14]:


def process_pred(ground_truth, prediction_tensor):
    n = tf.squeeze(prediction_tensor).numpy()
    df = pd.DataFrame(columns=['id_index','probability'])
    predictions_str = ''
    top_predictions = []
    index = 0
    for x in n:
        if x > cfg.PREDICTIONS_THRESHOLD:
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

        print("ground_truth[i]=%s", str(ground_truth[i]))
        print("sorted_predictions=%d", str(sorted_predictions))
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


# In[15]:


def evaluate_classification(image):
    temp_input = tf.expand_dims(read_image(image), 0) 
    h = head_model(temp_input)
    s = tail_model(h)
    return s


# In[16]:


ds_val, ds_info = tfds.load(name="coco/2017", split=split_val, data_dir=data_dir, shuffle_files=False, download=False, with_info=True)
ds_val = ds_val.map(functools.partial(my_preprocess), num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_val = ds_val.map(expand_dims_for_single_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_val = ds_val.range(100)

# iterator = ds_val.make_one_shot_iterator()

ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)


# In[ ]:





# In[17]:


count = 0
max_test_images = max_tests

coco_image_dir = '/home/suphale/snehal_bucket/coco/raw-data/val2017/'

total_time = 0.0
df = pd.DataFrame(columns=['img_path','ground_truth', 'top_predict', 'Prediction', 'accuracy', 'top_1_accuracy', 'top_5_accuracy', 'precision', 'recall', 'time'])
ds_val = ds_val.take(max_tests)
for sample_img_batch, ground_truth, img_path in tqdm(ds_val):
    count += 1
    img_path = img_path.numpy().decode()

    tk.startRecord(img_path)
    tk.logTime(img_path, tk.E_START_CLIENT_PROCESSING)

    tensor_shape = len(ground_truth.get_shape().as_list())
    if(tensor_shape > 1):
        ground_truth = tf.squeeze(ground_truth,[0])

    ground_truth = list(set(ground_truth.numpy()))
    # print(ground_truth)

    # accuracy, top_1_accuracy,top_5_accuracy,precision,recall, top_predictions, predictions_str = process_predictions(sample_img_batch, ground_truth)

    if(test_number == test.STANDALONE):
        predictions,predictions_prob = evaluate_standalone(sample_img_batch, img_path)
    if(test_number == test.JPEG_TRANSFER):
        predictions,predictions_prob = evaluate_file_over_server(coco_image_dir + img_path)
    if(test_number == test.DECODED_IMAGE_TRANSFER):
        predictions,predictions_prob = evaluate_over_server(sample_img_batch, img_path)
    if(test_number == test.DECODED_IMAGE_TRANSFER_ZLIB):
        predictions,predictions_prob = evaluate_over_server(sample_img_batch, img_path,zlib_compression=True)
    if(test_number == test.SPLIT_LAYER_3):
        predictions,predictions_prob = evaluate_over_server_head_model(sample_img_batch, img_path)
    if(test_number == test.SPLIT_LAYER_3_ZLIB):
        predictions,predictions_prob = evaluate_over_server_head_model(sample_img_batch, img_path,zlib_compression=True)

    tk.logTime(img_path, tk.E_STOP_CLIENT_PROCESSING)

    accuracy, top_1_accuracy,top_5_accuracy,precision,recall, top_predictions, predictions_str = process_predictions(cfg, imagesInfo, ground_truth,predictions, predictions_prob)
    df = df.append(
        {'image':img_path, 
        'ground_truth':(str(imagesInfo.get_segmentation_texts(ground_truth))),
        'top_predict':str(top_predictions),
        'Prediction':predictions_str,
        'accuracy':accuracy,
        'top_1_accuracy':top_1_accuracy,
        'top_5_accuracy':top_5_accuracy,
        'precision':precision,
        'recall':recall,
        'time':0,
        },
        ignore_index = True)
    truth_str = ' '.join([str(elem) for elem in imagesInfo.get_segmentation_texts(ground_truth)])
    # Logger.debug_print("ground_truth  : %s" % (truth_str))
    # Logger.debug_print("Prediction    : %s" % (predictions_str))

    tk.finishRecord(img_path)

df.to_csv(cfg.temp_path + '/results_'+cfg.timestr+'.csv')
av_column = df.mean(axis=0)

Logger.milestone_print("----------------:")
Logger.milestone_print("Test scenario   : %d %s" % (test_number, test_scenarios[test_number]))
Logger.milestone_print("Image shape     : (%d %d)" % (h_image_height, h_image_width))
Logger.milestone_print("Max tests       : %d" % (max_tests))
Logger.milestone_print("accuracy        : %.2f" % (av_column.accuracy))
Logger.milestone_print("top_1_accuracy  : %.2f" % (av_column.top_1_accuracy))
Logger.milestone_print("top_5_accuracy  : %.2f" % (av_column.top_5_accuracy))
Logger.milestone_print("precision       : %.2f" % (av_column.precision))
Logger.milestone_print("recall          : %.2f" % (av_column.recall))
Logger.milestone_print("time            : %.2f" % (av_column.time))

# tk.printAll()
tk.summary()


# In[18]:


ds_info


# In[ ]:



