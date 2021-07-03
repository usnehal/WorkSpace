#!/usr/bin/env python
# coding: utf-8

# In[1]:


import socket
import threading
import os
import json

from Config import Config


# In[2]:


class Server:
    def __init__(self,cfg):
        self.cfg = cfg
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
            c, addr = self.s.accept()
            print(c)

            threading.Thread(target=self.handle_client,args=(c,addr,)).start()

    def handle_client(self,c,addr):
        received_data = c.recv(1024).decode()
        print("received_data="+received_data)
        obj = json.loads(received_data)
        print(obj)
        c.send("OK".encode())

        write_name = 'test' + '.recd'
        if os.path.exists(write_name): os.remove(write_name)

        with open(write_name,'wb') as file:
            while 1:
                data = c.recv(1024)
                if not data:
                    break
                file.write(data)
            file.close()
        print(write_name,'successfully downloaded.')


# In[3]:


cfg = Config()
server = Server(cfg)


# In[ ]:




