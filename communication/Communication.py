import socket
import Logger

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