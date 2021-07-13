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

from Config import Config
import Logger
from Communication import Server
import Util


# In[3]:


class Encoder(Model):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        # build your Dense layer with relu activation
        self.dense = tf.keras.layers.Dense(embed_dim, activation='relu')
        
    def call(self, features):
        # extract the features from the image shape: (batch, 8*8, embed_dim)
        features = self.dense(features)
        return features    


# In[4]:


class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.units=units
        # build your Dense layer
        self.W1 = tf.keras.layers.Dense(units)
        # build your Dense layer
        self.W2 = tf.keras.layers.Dense(units)
        # build your final Dense layer with unit 1
        # self.V = tf.keras.layers.Dense(1, activation='softmax')
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features shape: (batch_size, 8*8, embedding_dim)
        # hidden shape: (batch_size, hidden_size)
        
        # Expand the hidden shape to shape: (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # build your score funciton to shape: (batch_size, 8*8, units)
        score = tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # extract your attention weights with shape: (batch_size, 8*8, 1)
        score = self.V(score)
        attention_weights = softmax(score, axis=1)

        # shape: create the context vector with shape (batch_size, 8*8,embedding_dim)
        context_vector = attention_weights * features
        # reduce the shape to (batch_size, embedding_dim)
        # context_vector = tf.reduce_sum(context_vector, axis=1)
        context_vector = tf.reduce_mean(context_vector, axis=1)

        return context_vector, attention_weights   


# In[5]:


class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.units = units
        # iniitalise your Attention model with units
        self.attention = Attention_model(self.units)
        # build your Embedding layer
        self.embed = tf.keras.layers.Embedding(vocab_size, self.cfg.embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        # build your Dense layer
        self.d1 = tf.keras.layers.Dense(self.units)
        # build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size)


    def call(self,x,features, hidden):
        #create your context vector & attention weights from attention model
        context_vector, attention_weights = self.attention(features, hidden)
        # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = self.embed(x)
        # Concatenate your input with the context vector from attention layer. 
        # Shape: (batch_size, 1, embedding_dim + embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)
        # Extract the output & hidden state from GRU layer. 
        # Output shape : (batch_size, max_length, hidden_size)
        output, state = self.gru(embed)
        output = self.d1(output)
        # shape : (batch_size * max_length, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2])) 
        # shape : (batch_size * max_length, vocab_size)
        output = self.d2(output)

        return output, state, attention_weights

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


# In[6]:


class TailModel:
    def __init__(self,cfg):
        self.cfg = cfg

        with open(self.cfg.saved_model_path + '/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        with open(self.cfg.saved_model_path + '/image_features_extract_model.json', 'r') as json_file:
            json_savedModel= json_file.read()
        self.image_features_extract_model = tf.keras.models.model_from_json(json_savedModel)
        self.image_features_extract_model.load_weights(self.cfg.saved_model_path + '/image_features_extract_model.h5')
        vocab_size = cfg.max_tokenized_words + 1

        s = tf.zeros([32, 64, 2048], tf.int32)

        self.encoder=Encoder(self.cfg.embedding_dim)

        self.decoder=Decoder(self.cfg.embedding_dim, self.cfg.units, vocab_size,cfg)

        features = self.encoder(s)

        hidden = self.decoder.init_state(batch_size=self.cfg.batch_size)
        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * self.cfg.batch_size, 1)

        predictions, hidden_out, attention_weights= self.decoder(dec_input, features, hidden)

        self.decoder.load_weights(self.cfg.saved_model_path + "/decoder.h5")
        self.encoder.load_weights(self.cfg.saved_model_path + "/encoder.h5")

    def evaluate(self,image):
        attention_features_shape = 64
        attention_plot = np.zeros((self.cfg.MAX_SEQ_LENGTH, attention_features_shape))

        hidden = self.decoder.init_state(batch_size=1)

        # process the input image to desired format before extracting features
        temp_input = tf.expand_dims(image, 0) 
        # Extract features using our feature extraction model
        img_tensor_val = self.extract_image_features(temp_input)
        # img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        # extract the features by passing the input to encoder
        features = self.encoder(img_tensor_val)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.cfg.MAX_SEQ_LENGTH):
            # get the output from decoder
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            #extract the predicted id(embedded value) which carries the max value
            predicted_id = tf.argmax(tf.transpose(predictions))
            predicted_id = predicted_id.numpy()[0]
            # map the id to the word from tokenizer and append the value to the result list
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot,predictions

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot,predictions

    def process_image_file(self,msg,shape):
        temp_file = '/tmp/temp.bin'
        f = open(temp_file, "wb")
        f.write(msg)
        f.close()

        image_tensor,label = Util.read_image(temp_file,[])

        t0 = time.perf_counter()
        result, attention_plot,pred_test  = self.evaluate(image_tensor)
        t1 = time.perf_counter() - t0
        pred_caption=' '.join(result).rsplit(' ', 1)[0]

        send_json_dict = {}
        send_json_dict['response'] = 'OK'
        send_json_dict['pred_caption'] = pred_caption
        send_json_dict['tail_model_time'] = t1

        app_json = json.dumps(send_json_dict)

        return str(app_json)

    def process_image_tensor(self,msg,shape):
        generated_np_array = np.frombuffer(msg, dtype=float32)
        generated_np_array = np.frombuffer(generated_np_array, dtype=float32)
        generated_image_np_array = generated_np_array.reshape(shape)
        image_tensor = tf.convert_to_tensor(generated_image_np_array, dtype=tf.float32)

        t0 = time.perf_counter()
        result, attention_plot,pred_test  = self.evaluate(image_tensor)
        t1 = time.perf_counter() - t0
        pred_caption=' '.join(result).rsplit(' ', 1)[0]

        send_json_dict = {}
        send_json_dict['response'] = 'OK'
        send_json_dict['pred_caption'] = pred_caption
        send_json_dict['tail_model_time'] = t1

        app_json = json.dumps(send_json_dict)

        return str(app_json)
        
    def extract_image_features(self, sample_img_batch):
        features = self.image_features_extract_model(sample_img_batch)
        features = tf.reshape(features, [sample_img_batch.shape[0],8*8, 2048])
        return features


# In[7]:


cfg = Config(None)

with open(cfg.saved_model_path + '/image_features_extract_model.json', 'r') as json_file:
    json_savedModel= json_file.read()
image_features_extract_model = tf.keras.models.model_from_json(json_savedModel)
image_features_extract_model.load_weights(cfg.saved_model_path + '/image_features_extract_model.h5')


# In[8]:


image_features_extract_model.summary()


# In[ ]:


image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')

# write code here to get the input of the image_model
new_input = image_model.input 
# write code here to get the output of the image_model
hidden_layer = image_model.output 

#build the final model using both input & output layer
image_features_extract_model = tf.keras.Model(inputs=new_input, outputs=hidden_layer)


# In[7]:


cfg = Config(None)
tailModel = TailModel(cfg)
server = Server(cfg, tailModel)
server.register_callback('data',tailModel.process_image_tensor)
server.register_callback('file',tailModel.process_image_file)
server.accept_connections()


# In[ ]:


server.callbacks


# In[ ]:




