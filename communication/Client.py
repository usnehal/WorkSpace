#!/usr/bin/env python
# coding: utf-8

# In[6]:


import socket
import os
import json
import  time

from Config import Config
import Util


# In[7]:


class Client:
    def __init__(self,cfg):
        self.cfg = cfg
        # self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.connect_to_server()

    def connect_to_server(self):
        # self.s.connect((self.cfg.server_ip,int(self.cfg.server_port)))
        # self.main()
        print("")

    def reconnect(self):
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.s.connect((self.cfg.server_ip,int(self.cfg.server_port)))

    def send_data(self,data_info, data_buffer):
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.s.connect((self.cfg.server_ip,int(self.cfg.server_port)))
        self.s.send(data_info.encode())

        confirmation = self.s.recv(1024)
        if confirmation.decode() == "OK":
            print('Sending data')
            self.s.sendall(data_buffer)

            print(file_name,'successfully sent.')

            self.s.shutdown(socket.SHUT_RDWR)
            self.s.close()
            # self.reconnect()
        else:
            print("Received error from server, %s" % (confirmation.decode()))


# In[8]:


cfg = Config()
client = Client(cfg)
file_name = '/home/suphale/WorkSpace/000000350497.jpg'


# In[ ]:





# In[9]:


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


# In[10]:


data_buffer,caption = Util.read_image(file_name,'')
send_json_dict = {}
send_json_dict['data_type'] = 'data'
send_json_dict['file_name'] = file_name
send_json_dict['data_size'] = (len(data_buffer))
# send_json_dict['data_buffer'] = blob_data
app_json = json.dumps(send_json_dict)
print(str(app_json))

t0= time.perf_counter()
client.send_data(str(app_json), data_buffer)
t1 = time.perf_counter() - t0

print("Time to send data: %.3f" % (t1))


# In[ ]:




