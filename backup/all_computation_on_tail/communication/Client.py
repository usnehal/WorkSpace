#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[7]:


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server', action='store', type=str, required=False)
args, unknown = parser.parse_known_args()
print(args.server)

server_ip = args.server


# In[ ]:


class Client:
    def __init__(self,cfg):
        self.cfg = cfg
        # self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    def reconnect(self):
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.s.connect((self.cfg.server_ip,int(self.cfg.server_port)))

    def send_data(self,data_info, data_buffer):
        Logger.debug_print("send_data:Entry")
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        Logger.debug_print("send_data:Connect")
        self.s.connect((self.cfg.server_ip,int(self.cfg.server_port)))
        Logger.debug_print("send_data:send data_info")
        self.s.send(data_info.encode())

        Logger.debug_print("send_data:receive response")
        confirmation = self.s.recv(1024).decode()
        Logger.debug_print("send_data:confirmation = "+confirmation)
        if confirmation == "OK":
            Logger.debug_print('send_data:Sending data')
            self.s.sendall(data_buffer)

            Logger.debug_print('send_data:successfully sent data.')

            pred_caption = self.s.recv(1024)

            Logger.debug_print('send_data:received '+pred_caption.decode())
            self.s.shutdown(socket.SHUT_RDWR)
            self.s.close()
            Logger.debug_print(pred_caption.decode())
            return pred_caption.decode()
            # self.reconnect()
        else:
            print("Received error from server, %s" % (confirmation.decode()))


# In[ ]:


cfg = Config(server_ip)
client = Client(cfg)


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

    # send_json_dict['data_buffer'] = blob_data
    app_json = json.dumps(send_json_dict)
    # print(str(app_json))

    t0= time.perf_counter()
    pred_caption = client.send_data(str(app_json), byte_buffer_to_send)
    t1 = time.perf_counter() - t0

    # print("Time to send data: %.3f" % (t1))

    return pred_caption, [], []


# In[ ]:


def filt_text(text):
    filt=['<start>','<unk>','<end>'] 
    temp= text.split()
    [temp.remove(j) for k in filt for j in temp if k==j]
    text=' '.join(temp)
    return text

all_imgs = [cfg.images_path + line.rstrip() for line in open(cfg.list_file)]
all_imgs = sorted(all_imgs)
total_num_images = len(all_imgs)
Logger.debug_print("The total images present in the dataset: {}".format(total_num_images))

#Visualise both the images & text present in the dataset
Logger.debug_print("The total images present in the dataset: {}".format(total_num_images))
num_lines = sum(1 for line in open(cfg.text_file))
Logger.debug_print("The total number of lines in the caption file: {}".format(num_lines))

# define a function to clean text data
def extract_jpg_caption(line):
    char_filter = r"[^\w]"

    jpg_path = None
    caption = None

    jpg_position = line.find(".jpg")
    if(jpg_position != -1):
        jpg_path = cfg.images_path + '/' + line[:jpg_position+4]

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

    #store all the image id here
all_img_id= [] 
#store all the image path here
all_img_vector=[]
#store all the captions here
annotations_dict = {} 
annotations= [] 
# list of all captions in word list format
annotations_word_list = []

def load_doc(filename):
    #your code here
    file  = open(filename, 'r') 
    Lines = file.readlines() 
    
    text = ""
    count = 0
    for line in Lines:
        jpg_path, caption = extract_jpg_caption(line)
        if(jpg_path != None):
            all_img_id.append(count)
            all_img_vector.append(jpg_path)
            annotations.append(caption)

            caption_list = []
            if jpg_path in annotations_dict.keys():
                caption_list = annotations_dict[jpg_path]
            caption_list.append(caption)
            annotations_dict[jpg_path] = caption_list

            word_list = caption.split()
            annotations_word_list.append(word_list)
            text += " " + caption
            count += 1
    file.close()
    return text

doc = load_doc(cfg.text_file)

total_time = 0.0
max_test_images = cfg.total_test_images
for i in range(max_test_images):
    print("")
    random_num = random.randint(0,total_num_images-1)
    img_path = all_img_vector[random_num]
    # image = io.imread(img_path)
    # plt.imshow(image)

    test_image = img_path
    real_caption = annotations[random_num]

    t0= time.perf_counter()
    result, attention_plot,pred_test = evaluate_over_server(test_image)
    t1 = time.perf_counter() - t0
    total_time = total_time + t1

    real_caption=filt_text(real_caption)      

    pred_caption = result
    # pred_caption=' '.join(result).rsplit(' ', 1)[0]

    # real_appn = []
    # real_appn.append(real_caption.split())
    # reference = real_appn
    # candidate = pred_caption.split()

    real_appn = []
    real_caption_list = annotations_dict[img_path]
    for real_caption in real_caption_list:
        real_caption=filt_text(real_caption)
        real_appn.append(real_caption.split())
    reference = real_appn
    candidate = pred_caption.split()

    score = sentence_bleu(reference, candidate, weights=[1]) #set your weights)

    print("Time: %.2f BLEU: %.2f" % (t1,score))
    print ('Real:', real_caption)
    print ('Pred:', pred_caption)

print("Average time = %f" % (total_time/max_test_images))

