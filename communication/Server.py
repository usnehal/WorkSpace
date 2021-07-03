#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import socket
import threading
import os
import json
import numpy as np
import tensorflow as tf
from   numpy import float32
from   tensorflow.keras import layers,Model
import pickle5 as pickle
from   tensorflow.keras.preprocessing.text import Tokenizer
from   tensorflow.keras.activations import tanh
from   tensorflow.keras.activations import softmax

from Config import Config


# In[ ]:


class Server:
    def __init__(self,cfg,tailModel):
        self.cfg = cfg
        self.tailModel = tailModel
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.accept_connections()
    
    def accept_connections(self):
        ip = '' 
        port = self.cfg.server_port

        print('Running on IP: '+ip)
        print('Running on port: '+str(port))

        self.s.bind((ip,port))
        self.s.listen(100)

        while 1:
            try:
                c, addr = self.s.accept()
            except KeyboardInterrupt as e:
                print("ctrl+c,Exiting gracefully")
                self.s.shutdown(socket.SHUT_RDWR)
                self.s.close()
                exit(0)
            # print(c)

            threading.Thread(target=self.handle_client,args=(c,addr,)).start()

    def handle_client(self,c,addr):
        received_data = c.recv(1024).decode()
        print("received_data="+received_data)
        obj = json.loads(received_data)
        print(obj)
        tensor_shape = obj['data_shape']
        c.send("OK".encode())

        write_name = 'test' + '.recd'
        if os.path.exists(write_name): os.remove(write_name)

        total_data = 0
        msg = bytearray()
        while 1:
            data = c.recv(1024)
            # print(type(data))
            msg.extend(data)
            if not data:
                break
            total_data += len(data)
        
        print('total size of msg=%d' % (len(msg)))
        
        generated_np_array = np.frombuffer(msg, dtype=float32)

        generated_image_np_array = generated_np_array.reshape(tensor_shape)
        generate_image_tensor = tf.convert_to_tensor(generated_image_np_array, dtype=tf.float32)
        print(generate_image_tensor.shape)
        result, attention_plot,pred_test  = tailModel.evaluate(generate_image_tensor)
        pred_caption=' '.join(result).rsplit(' ', 1)[0]
        # candidate = pred_caption.split()
        print ('Pred:', pred_caption)


# In[ ]:


class Encoder(Model):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        # build your Dense layer with relu activation
        self.dense = tf.keras.layers.Dense(embed_dim, activation='relu')
        
    def call(self, features):
        # extract the features from the image shape: (batch, 8*8, embed_dim)
        features = self.dense(features)
        return features    


# In[ ]:


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


# In[ ]:


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


# In[ ]:


class TailModel:
    def __init__(self,cfg):
        self.cfg = cfg

        with open('saved_model/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        with open('saved_model/image_features_extract_model.json', 'r') as json_file:
            json_savedModel= json_file.read()
        self.image_features_extract_model = tf.keras.models.model_from_json(json_savedModel)
        self.image_features_extract_model.load_weights('saved_model/image_features_extract_model.h5')
        vocab_size = cfg.max_tokenized_words + 1

        s = tf.zeros([32, 64, 2048], tf.int32)

        self.encoder=Encoder(self.cfg.embedding_dim)

        self.decoder=Decoder(self.cfg.embedding_dim, self.cfg.units, vocab_size,cfg)

        features = self.encoder(s)

        hidden = self.decoder.init_state(batch_size=self.cfg.batch_size)
        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * self.cfg.batch_size, 1)

        predictions, hidden_out, attention_weights= self.decoder(dec_input, features, hidden)

        self.decoder.load_weights("saved_model/decoder.h5")
        self.encoder.load_weights("saved_model/encoder.h5")

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

    def extract_image_features(self, sample_img_batch):
        print("extract_image_features shape=" + str(tf.shape(sample_img_batch)))
        features = self.image_features_extract_model(sample_img_batch)
        features = tf.reshape(features, [sample_img_batch.shape[0],8*8, 2048])
        return features


# In[ ]:


cfg = Config()
tailModel = TailModel(cfg)
server = Server(cfg, tailModel)


# In[ ]:




