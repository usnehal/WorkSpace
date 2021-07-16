import tensorflow as tf
import re
import time
from tabulate import tabulate
import pandas as pd
import pandas as pd
import os
import pickle5 as pickle
import socket
from   numpy import float32
import socket
import threading
import json
import numpy as np
import tensorflow as tf
import zlib
from    sklearn.metrics import accuracy_score

import  socket

class test:
    STANDALONE = 0
    JPEG_TRANSFER = 1
    DECODED_IMAGE_TRANSFER = 2
    DECODED_IMAGE_TRANSFER_ZLIB = 3
    SPLIT_LAYER_3 = 4
    SPLIT_LAYER_3_ZLIB = 5

class Config():
    def __init__(self, server_ip=None):
        self.host = None
        self.HOST_NIMBLE = 1
        self.HOST_WSL = 2
        self.HOST_PI = 3

        self.total_test_images = 100
        self.max_tokenized_words = 20000
        self.MAX_SEQ_LENGTH = 25
        self.batch_size = 32
        self.embedding_dim = 256 
        self.units = 512
        if(server_ip != None):
            self.server_ip = server_ip
        else:
            self.server_ip = 'localhost'
        # self.server_ip = '35.200.232.85'
        self.server_port = 5002

        host = socket.gethostname()
        self.workspace_path = '/home/suphale/WorkSpace' 
        self.images_path = None
        if('cuda' in host):
            self.host = self.HOST_NIMBLE
            self.workspace_path='/mnt/disks/user/project/WorkSpace'
            print("In NimbleBox")
        if(host == 'LTsuphale-NC2JM'):
            self.host = self.HOST_WSL
            print("In WSL")
        if(host == 'raspberrypi'):
            self.host = self.HOST_PI
            self.workspace_path='/home/pi/WorkSpace'
            print("In raspberry-pi")
        self.images_path = self.workspace_path + '/test_images'
        self.saved_model_path = self.workspace_path + '/saved_model'
        self.temp_path = self.workspace_path + '/temp'

        self.text_file = self.workspace_path + '/lists/captions_' + str(self.total_test_images) + '.txt'
        self.list_file = self.workspace_path + '/lists/images_' + str(self.total_test_images) + '.txt'
        self.instances_file = self.workspace_path + '/lists/instances_' + str(self.total_test_images) + '.txt'
        self.df_captions_csv = self.workspace_path + '/lists/df_captions_' + str(self.total_test_images) + '.csv'
        self.df_instances_csv = self.workspace_path + '/lists/df_instances_' + str(self.total_test_images) + '.csv'
        self.df_categories_csv = self.workspace_path + '/lists/df_categories.csv'

        self.timestr = time.strftime("%Y_%m_%d-%H_%M")

        self.PREDICTIONS_THRESHOLD = 0.4

    def __getitem__(self):

        return self.x

    def __len__(self):
        return len(self.x)

cfg = Config()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

debug_level = 0
class Logger:
    def __init__(self):
        debugLogs = False
        debug_level = 1
    def set_log_level(level):
        global debug_level
        debug_level = level

    def get_log_level(level):
        return debug_level

    def debug_print(str):
        if(debug_level >= 2):
            print(str)

    def event_print(str):
        if(debug_level >= 1):
            print(bcolors.OKCYAN + str + bcolors.ENDC)

    def milestone_print(str):
        if(debug_level >= 0):
            print(bcolors.OKGREEN + str + bcolors.ENDC)
            file1 = open(cfg.temp_path + "/log.txt", "a")  # append mode
            file1.write(str + "\n")
            file1.close()

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



class TimeKeeper:
    def __init__(self):
        self.pretty_df = pd.DataFrame(columns=['Image','Total_Time','Comm_Time'])

        self.E_START_CLIENT_PROCESSING = 'E_START_CLIENT_PROCESSING'
        self.E_STOP_CLIENT_PROCESSING = 'E_STOP_CLIENT_PROCESSING'
        self.E_START_COMMUNICATION = 'E_START_COMMUNICATION'
        self.E_STOP_COMMUNICATION = 'E_STOP_COMMUNICATION'

        self.I_BUFFER_SIZE = 'I_BUFFER_SIZE'
        self.I_REAL_CAPTION = 'I_REAL_CAPTION'
        self.I_PRED_CAPTION = 'I_PRED_CAPTION'
        self.I_CLIENT_PROCESSING_TIME = 'I_CLIENT_PROCESSING_TIME'
        self.I_COMMUNICATION_TIME = 'I_COMMUNICATION_TIME'
        self.I_TAIL_MODEL_TIME = 'I_TAIL_MODEL_TIME'

        self.records = {}

    def startRecord(self, image):
        self.records[image] = {}

    def logTime(self, image, event):
        self.records[image][event] = time.perf_counter()

    def logInfo(self, image, info_key, info):
        self.records[image][info_key] = info

    def finishRecord(self, image):
        # self.records[image] = {}
        self.records[image][self.I_CLIENT_PROCESSING_TIME] = self.records[image][self.E_STOP_CLIENT_PROCESSING] - \
            self.records[image][self.E_START_CLIENT_PROCESSING]
        self.records[image][self.I_COMMUNICATION_TIME] = self.records[image][self.E_STOP_COMMUNICATION] - \
            self.records[image][self.E_START_COMMUNICATION]
        record = self.records[image]
        pretty_record = {}
        pretty_record['Image'] = image.rsplit('/', 1)[-1]
        pretty_record['Total_Time'] = "{:.02f}".format(self.records[image][self.I_CLIENT_PROCESSING_TIME])
        pretty_record['Comm_Time'] = "{:.02f}".format(self.records[image][self.I_COMMUNICATION_TIME])
        self.pretty_df = self.pretty_df.append(pretty_record,ignore_index=True)
        pass

    def printAll(self):
        Logger.event_print(tabulate(self.pretty_df, headers='keys', tablefmt='psql'))

    def summary(self):
        df = pd.DataFrame(self.records)
        df_t = df.T
        
        # df_t.to_csv("TimeKeeper.csv")
        average_inference_time = df_t[self.I_CLIENT_PROCESSING_TIME].mean()
        average_head_model_time = df_t[self.I_CLIENT_PROCESSING_TIME].mean() - df_t[self.I_COMMUNICATION_TIME].mean()
        average_communication_time = df_t[self.I_COMMUNICATION_TIME].mean() - df_t[self.I_TAIL_MODEL_TIME].mean()
        average_tail_model_time = df_t[self.I_TAIL_MODEL_TIME].mean()
        average_communication_payload = int(df_t[self.I_BUFFER_SIZE].mean())
        Logger.milestone_print("Avg total time  : %.2f s" % (average_inference_time))
        Logger.milestone_print("Avg head time   : %.2f s" % (average_head_model_time))
        Logger.milestone_print("Avg network time: %.2f s" % (average_communication_time))
        Logger.milestone_print("Avg tail time   : %.2f s" % (average_tail_model_time))
        Logger.milestone_print("Avg nw payload  : " + f"{int(average_communication_payload):,d}")

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
        self.df_instances = pd.read_csv(self.cfg.df_instances_csv)
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

class Client:
    def __init__(self,cfg):
        self.cfg = cfg
        # self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    def connect(self):
        Logger.debug_print("connect:Entry")
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        Logger.debug_print("connect:Connect")
        self.s.connect((self.cfg.server_ip,int(self.cfg.server_port)))

    def disconnect(self):
        Logger.debug_print("disconnect:Entry")
        self.s.shutdown(socket.SHUT_RDWR)
        Logger.debug_print("disconnect:Connect")
        self.s.close()

    def reconnect(self):
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.s.connect((self.cfg.server_ip,int(self.cfg.server_port)))

    def send_load_model_request(self,data_info):
        self.connect()
        Logger.debug_print("send_data:send data_info")
        self.s.send(data_info.encode())

        Logger.debug_print("send_data:receive response")
        confirmation = self.s.recv(1024).decode()
        Logger.debug_print("send_data:confirmation = "+confirmation)
        if confirmation == "OK":
            Logger.debug_print('send_data:Sending data')
            return "OK"
        else:
            print("Received error from server, %s" % (confirmation.decode()))
            return "Error"

    def send_data(self,data_info, data_buffer):
        self.connect()
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

class Server:
    def __init__(self,cfg,tailModel):
        self.cfg = cfg
        self.tailModel = tailModel
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.request_count = 0
        self.callbacks = {}

        # self.accept_connections()

    def register_callback(self, obj, callback):
        print('register_callback obj='+obj)
        if obj not in self.callbacks:
            self.callbacks[obj] = None
        self.callbacks[obj] = callback
        print('register_callback self.callbacks=%s' % (str(self.callbacks)))
    
    def accept_connections(self):
        ip = '' 
        port = self.cfg.server_port

        Logger.milestone_print('Running on IP: '+ip)
        Logger.milestone_print('Running on port: '+str(port))

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
        # global request_count
        # print(addr)
        # print('%d' % (self.request_count), end ="\r") 
        self.request_count = self.request_count + 1
        Logger.debug_print("handle_client:Entry")
        received_data = c.recv(1024).decode()
        Logger.debug_print("handle_client:received_data="+received_data)
        obj = json.loads(received_data)
        Logger.debug_print(obj)
        data_type = obj['data_type']
        if(data_type == 'load_model_request'):
            model_type = obj['model']
            response = ''
            if data_type in self.callbacks :
                callback = self.callbacks[data_type]
                response = callback(model_type,None)

            Logger.debug_print("handle_client:sending pred_caption" + response)
            c.send(response.encode())
            # candidate = pred_caption.split()
            Logger.debug_print ('response:' + response)
        else:
            tensor_shape = obj['data_shape']
            if 'zlib_compression' in obj.keys():
                zlib_compression = obj['zlib_compression']
            else:
                zlib_compression = False
            Logger.debug_print("handle_client:sending OK")
            c.send("OK".encode())

            max_data_to_be_received = obj['data_size']
            total_data = 0
            msg = bytearray()
            while 1:
                # print("handle_client:calling recv total_data=%d data_size=%d" % (total_data, max_data_to_be_received))
                if(total_data >= max_data_to_be_received):
                    Logger.debug_print("handle_client:received all data")
                    break
                data = c.recv(1024)
                # print(type(data))
                msg.extend(data)
                if not data:
                    Logger.debug_print("handle_client:while break")
                    break
                total_data += len(data)
            
            Logger.debug_print('total size of msg=%d' % (len(msg)))
            
            if(zlib_compression == 'yes'):
                msg = zlib.decompress(msg)

            response = ''
            if data_type in self.callbacks :
                callback = self.callbacks[data_type]
                response = callback(msg,tensor_shape)

            Logger.debug_print("handle_client:sending pred_caption" + response)
            c.send(response.encode())
            # candidate = pred_caption.split()
            Logger.debug_print ('response:' + response)

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
