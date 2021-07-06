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

from   Config import Config
import Util
import Logger
from   Communication import Client
from ImagesInfo import ImagesInfo
from TimeKeeper import TimeKeeper


# In[3]:


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
    test_number = 3

test_scenarios = {1:"Complete jpg file buffer transfer", 
                    2:"Decoded image buffer transfer",
                    3:"Decoded image buffer transfer with zlib compression"}

print("Test scenario = %d %s" % (test_number, test_scenarios[test_number]))


# In[4]:


Logger.set_log_level(verbose)
tk = TimeKeeper()
cfg = Config(server_ip)
client = Client(cfg)
imagesInfo = ImagesInfo(cfg)


# In[5]:


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

        pred_caption = response['pred_caption']
        tail_model_time = response['tail_model_time']
        tk.logInfo(img_path, tk.I_TAIL_MODEL_TIME, tail_model_time)

        return pred_caption, [], []


# In[6]:


# tf.compat.v1.disable_eager_execution()


# In[7]:


def evaluate_over_server(file_name, zlib_compression=False):
    image_tensor,caption = Util.read_image(file_name,'')

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

    pred_caption = response['pred_caption']
    tail_model_time = response['tail_model_time']
    tk.logInfo(img_path, tk.I_TAIL_MODEL_TIME, tail_model_time)

    return pred_caption, [], []


# In[8]:



total_time = 0.0
max_test_images = cfg.total_test_images
for i in range(max_test_images):
    Logger.event_print("")
    random_num = random.randint(0,max_test_images-1)
    img_path = imagesInfo.getImagePath(random_num)
    # image = io.imread(img_path)
    # plt.imshow(image)

    real_caption = imagesInfo.getCaption(random_num)

    tk.startRecord(img_path)
    tk.logTime(img_path, tk.E_START_CLIENT_PROCESSING)

    if(test_number == 1):
        pred_caption, attention_plot,pred_test = evaluate_file_over_server(img_path)
    if(test_number == 2):
        pred_caption, attention_plot,pred_test = evaluate_over_server(img_path)
    if(test_number == 3):
        pred_caption, attention_plot,pred_test = evaluate_over_server(img_path,zlib_compression=True)

    tk.logTime(img_path, tk.E_STOP_CLIENT_PROCESSING)

    real_caption=Util.filt_text(real_caption)      

    reference = imagesInfo.getAllCaptions(img_path)
    candidate = pred_caption.split()

    score = sentence_bleu(reference, candidate, weights=[1]) #set your weights)

    tk.logInfo(img_path, tk.I_BLEU, score)
    tk.logInfo(img_path, tk.I_REAL_CAPTION, real_caption)
    tk.logInfo(img_path, tk.I_PRED_CAPTION, pred_caption)
    tk.finishRecord(img_path)

    Logger.event_print("BLEU: %.2f" % (score))
    Logger.event_print ('Real: %s' % (real_caption))
    Logger.event_print ('Pred: %s' % (pred_caption))

tk.printAll()
tk.summary()


# In[ ]:




