import socket
import Logger
from   numpy import float32
import socket
import threading
import json
import numpy as np
import tensorflow as tf
import zlib

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
        # global request_count
        # print(addr)
        print('%d' % (self.request_count), end ="\r") 
        self.request_count = self.request_count + 1
        Logger.debug_print("handle_client:Entry")
        received_data = c.recv(1024).decode()
        Logger.debug_print("handle_client:received_data="+received_data)
        obj = json.loads(received_data)
        Logger.debug_print(obj)
        data_type = obj['data_type']
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
        # result, attention_plot,pred_test  = tailModel.evaluate(generate_image_tensor)
        # pred_caption=' '.join(result).rsplit(' ', 1)[0]

        Logger.debug_print("handle_client:sending pred_caption" + response)
        c.send(response.encode())
        # candidate = pred_caption.split()
        Logger.debug_print ('response:' + response)
        