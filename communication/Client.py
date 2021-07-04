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

from   Config import Config
import Util
import Logger
from   Communication import Client
from ImagesInfo import ImagesInfo
from TimeKeeper import TimeKeeper


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server', action='store', type=str, required=False)
args, unknown = parser.parse_known_args()
print(args.server)

server_ip = args.server


# In[ ]:


tk = TimeKeeper()
cfg = Config(server_ip)
client = Client(cfg)
imagesInfo = ImagesInfo(cfg)


# In[ ]:


if False:
    with open(file_name, 'rb') as file_t:
        blob_data = bytearray(file_t.read())
        send_json_dict = {}
        send_json_dict['data_type'] = 'file'
        send_json_dict['file_name'] = file_name
        send_json_dict['data_size'] = (len(blob_data))
        # send_json_dict['data_buffer'] = blob_data
        app_json = json.dumps(send_json_dict)
        print(str(app_json))

        t0= time.perf_counter()
        client.send_data(str(app_json), blob_data)
        t1 = time.perf_counter() - t0

        print("Time to send file: %.3f" % (t1))
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

        pred_caption = client.send_data(str(app_json), byte_buffer_to_send)

        tk.logTime(img_path, tk.E_STOP_COMMUNICATION)

        return pred_caption, [], []


# In[ ]:


# tf.compat.v1.disable_eager_execution()


# In[ ]:


def evaluate_over_server(file_name):
    image_tensor,caption = Util.read_image(file_name,'')

    image_np_array = image_tensor.numpy()

    byte_buffer_to_send = image_np_array.tobytes()
    type(byte_buffer_to_send)

    send_json_dict = {}
    send_json_dict['data_type'] = 'data'
    send_json_dict['file_name'] = file_name
    send_json_dict['data_size'] = (len(byte_buffer_to_send))
    send_json_dict['data_shape'] = image_np_array.shape

    app_json = json.dumps(send_json_dict)

    tk.logInfo(img_path, tk.I_BUFFER_SIZE, len(byte_buffer_to_send))

    tk.logTime(img_path, tk.E_START_COMMUNICATION)

    pred_caption = client.send_data(str(app_json), byte_buffer_to_send)

    tk.logTime(img_path, tk.E_STOP_COMMUNICATION)

    return pred_caption, [], []


# In[ ]:



total_time = 0.0
max_test_images = cfg.total_test_images
for i in range(max_test_images):
    print("")
    random_num = random.randint(0,max_test_images-1)
    img_path = imagesInfo.getImagePath(random_num)
    # image = io.imread(img_path)
    # plt.imshow(image)

    real_caption = imagesInfo.getCaption(random_num)

    tk.startRecord(img_path)
    tk.logTime(img_path, tk.E_START_CLIENT_PROCESSING)

    # pred_caption, attention_plot,pred_test = evaluate_over_server(img_path)
    pred_caption, attention_plot,pred_test = evaluate_file_over_server(img_path)

    tk.logTime(img_path, tk.E_STOP_CLIENT_PROCESSING)

    real_caption=Util.filt_text(real_caption)      

    reference = imagesInfo.getAllCaptions(img_path)
    candidate = pred_caption.split()

    score = sentence_bleu(reference, candidate, weights=[1]) #set your weights)

    tk.logInfo(img_path, tk.I_BLEU, score)
    tk.logInfo(img_path, tk.I_REAL_CAPTION, real_caption)
    tk.logInfo(img_path, tk.I_PRED_CAPTION, pred_caption)
    tk.finishRecord(img_path)

    print("BLEU: %.2f" % (score))
    print ('Real:', real_caption)
    print ('Pred:', pred_caption)

tk.printAll()
tk.summary()


# In[ ]:




