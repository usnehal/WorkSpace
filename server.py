#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3


# In[2]:


import os
import numpy as np
import tensorflow as tf
from   tensorflow.keras import layers,Model
import pickle5 as pickle
from   tensorflow.keras.preprocessing.text import Tokenizer
from   tensorflow.keras.activations import tanh
from   tensorflow.keras.activations import softmax
from   numpy import float32
from   numpy import byte
import json
import time
import zlib
import pickle5 as pickle

from common.config import Config
from common.logger import Logger
from common.communication import Client
from common.communication import Server
from common.helper import ImagesInfo 
from common.timekeeper import TimeKeeper
from common.helper import read_image, filt_text, get_predictions
from CaptionModel import CaptionModel


# In[3]:


class TailModel:
    def __init__(self,cfg):
        self.cfg = cfg
        self.model = None

    def evaluate(self,image):
        result = self.model(image)
        return result


# In[4]:


model = None
captionModel = None

def handle_load_model(msg,shape):
    global model
    global captionModel
    if(msg == 'model'):
        model_path = cfg.saved_model_path + '/model'
        Logger.milestone_print("Loading model : %s from %s" % (model,model_path))
        model = None
        model = tf.keras.models.load_model(model_path)
        # model = tf.keras.models.load_model(cfg.temp_path + '/extractor_model')
        return "OK"
    if(msg == 'captionModel'):
        Logger.milestone_print("Loading model : %s" % (msg))
        captionModel = None
        captionModel = CaptionModel()
        return "OK"
    if(msg == 'tail_model'):
        model_path = cfg.saved_model_path + '/tail_model'
        Logger.milestone_print("Loading model : %s from %s" % (model,model_path))
        model = None
        model = tf.keras.models.load_model(model_path)
        return "OK"


# In[5]:


def handle_image_file(msg,shape):
    temp_file = '/tmp/temp.bin'
    f = open(temp_file, "wb")
    f.write(msg)
    f.close()

    t0 = time.perf_counter()
    image_tensor = tf.expand_dims(read_image(temp_file), 0) 
    features, result = model(image_tensor)
    features = tf.reshape(features, [1,8*8, 2048])
    caption_tensor = captionModel.evaluate(features)
    t1 = time.perf_counter() - t0

    top_predictions, predictions_prob = get_predictions(cfg, result)

    send_json_dict = {}
    send_json_dict['response'] = 'OK'
    send_json_dict['predictions'] = top_predictions
    send_json_dict['predictions_prob'] = predictions_prob
    send_json_dict['predicted_captions'] = caption_tensor
    send_json_dict['tail_model_time'] = t1

    app_json = json.dumps(send_json_dict)

    return str(app_json)


# In[6]:


def handle_image_tensor(msg,shape):
    generated_np_array = np.frombuffer(msg, dtype=float32)
    generated_np_array = np.frombuffer(generated_np_array, dtype=float32)
    generated_image_np_array = generated_np_array.reshape(shape)
    image_tensor = tf.convert_to_tensor(generated_image_np_array, dtype=tf.float32)

    t0 = time.perf_counter()
    features, result = model(image_tensor)
    features = tf.reshape(features, [1,8*8, 2048])
    caption_tensor = captionModel.evaluate(features)
    t1 = time.perf_counter() - t0

    top_predictions, predictions_prob = get_predictions(cfg, result)

    send_json_dict = {}
    send_json_dict['response'] = 'OK'
    send_json_dict['predictions'] = top_predictions
    send_json_dict['predictions_prob'] = predictions_prob
    send_json_dict['predicted_captions'] = caption_tensor
    send_json_dict['tail_model_time'] = t1

    app_json = json.dumps(send_json_dict)

    return str(app_json)


# In[7]:


def extract_image_features(self, sample_img_batch):
    features = self.image_features_extract_model(sample_img_batch)
    features = tf.reshape(features, [sample_img_batch.shape[0],8*8, 2048])
    return features


# In[8]:


Logger.set_log_level(1)
# logger = Logger()
tk = TimeKeeper()
cfg = Config(None)
client = Client(cfg)
imagesInfo = ImagesInfo(cfg)


# In[9]:


tailModel = TailModel(cfg)
server = Server(cfg)
server.register_callback('data',handle_image_tensor)
server.register_callback('file',handle_image_file)
server.register_callback('load_model_request',handle_load_model)
server.accept_connections()


# In[ ]:




