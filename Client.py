#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3


# In[ ]:


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

from Helper import Config, ImagesInfo, Logger, Client, TimeKeeper
from Helper import read_image, filt_text, process_predictions


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server', action='store', type=str, required=False)
parser.add_argument('-t', '--test_number', action='store', type=int, required=False)
parser.add_argument('-v', '--verbose', action='store', type=int, required=False)
args, unknown = parser.parse_known_args()
print(args.server)

server_ip = args.server
test_number = args.test_number
verbose = args.verbose

if(verbose == None):
    verbose = 1

if(test_number == None):
    test_number = 4

test_scenarios = {  1:"Complete jpg file buffer transfer", 
                    2:"Decoded image buffer transfer",
                    3:"Decoded image buffer transfer with zlib compression",
                    4:"split model at layer 3",
                    5:"split model at layer 3 with zlib compression",
                    }

print("Test scenario = %d %s" % (test_number, test_scenarios[test_number]))


# In[ ]:


Logger.set_log_level(verbose)
tk = TimeKeeper()
cfg = Config(server_ip)
client = Client(cfg)
imagesInfo = ImagesInfo(cfg)

# client.connect()


# In[ ]:


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


# In[ ]:


# tf.compat.v1.disable_eager_execution()


# In[ ]:


def evaluate_over_server(file_name, zlib_compression=False):
    image_tensor = read_image(file_name)
    image_tensor = tf.expand_dims(image_tensor, 0) 

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


def evaluate_over_server_head_model(head_model, file_name, zlib_compression=False):

    temp_input = tf.expand_dims(read_image(file_name), 0) 
    intermediate_tensor = head_model(temp_input)
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


# In[ ]:


if(test_number in [1,2,3]):
    # head_model = tf.keras.models.load_model(cfg.saved_model_path + '/model')
    send_json_dict = {}
    send_json_dict['data_type'] = 'load_model_request'
    send_json_dict['model'] = '/model'
    app_json = json.dumps(send_json_dict)
    response = client.send_load_model_request(str(app_json))
    assert(response == 'OK')
if(test_number in [4,5]):
    head_model = tf.keras.models.load_model(cfg.saved_model_path + '/new_head_model')
    send_json_dict = {}
    send_json_dict['data_type'] = 'load_model_request'
    send_json_dict['model'] = '/new_tail_model'
    app_json = json.dumps(send_json_dict)
    response = client.send_load_model_request(str(app_json))
    assert(response == 'OK')

total_time = 0.0
max_test_images = cfg.total_test_images
df = pd.DataFrame(columns=['img_path','ground_truth', 'top_predict', 'Prediction', 'accuracy', 'top_1_accuracy', 'top_5_accuracy', 'precision', 'recall', 'time'])

for i in tqdm(range(max_test_images)):
    # Logger.event_print("")
    img_path = imagesInfo.getImagePath(i)
    # image = io.imread(img_path)
    # plt.imshow(image)
    ground_truth = imagesInfo.get_segmentation_id_indexes(img_path)

    tk.startRecord(img_path)
    tk.logTime(img_path, tk.E_START_CLIENT_PROCESSING)

    if(test_number == 1):
        predictions,predictions_prob = evaluate_file_over_server(img_path)
    if(test_number == 2):
        predictions,predictions_prob = evaluate_over_server(img_path)
    if(test_number == 3):
        predictions,predictions_prob = evaluate_over_server(img_path,zlib_compression=True)
    if(test_number == 4):
        predictions,predictions_prob = evaluate_over_server_head_model(head_model, img_path)
    if(test_number == 5):
        predictions,predictions_prob = evaluate_over_server_head_model(head_model, img_path,zlib_compression=True)

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

Logger.milestone_print("accuracy        : %.2f" % (av_column.accuracy))
Logger.milestone_print("top_1_accuracy  : %.2f" % (av_column.top_1_accuracy))
Logger.milestone_print("top_5_accuracy  : %.2f" % (av_column.top_5_accuracy))
Logger.milestone_print("precision       : %.2f" % (av_column.precision))
Logger.milestone_print("recall          : %.2f" % (av_column.recall))
Logger.milestone_print("time            : %.2f" % (av_column.time))

# tk.printAll()
tk.summary()


# In[ ]:




