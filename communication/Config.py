
import  socket

class Config():
    def __init__(self):
        self.host = None
        self.HOST_NIMBLE = 1
        self.HOST_WSL = 2
        self.HOST_PI = 3

        self.total_test_images = 128
        self.max_tokenized_words = 20000
        self.MAX_SEQ_LENGTH = 25
        self.batch_size = 32
        self.embedding_dim = 256 
        self.units = 512
        self.server_ip = 'localhost'
        # self.server_ip = '35.200.232.85'
        self.server_port = 5002

        host = socket.gethostname()
        if('cuda' in host):
            self.host = self.HOST_NIMBLE
            print("In NimbleBox")
        if(host == 'LTsuphale-NC2JM'):
            self.host = self.HOST_WSL
            print("In WSL")
        if(host == 'raspberrypi'):
            self.host = self.HOST_PI
            print("In raspberry-pi")

        self.images_path = None
        if(self.host == self.HOST_WSL):
            self.images_path='/home/suphale/WorkSpace/test_images'
        if(self.host == self.HOST_NIMBLE):
            self.images_path='/mnt/disks/user/project/coco/train2017/'
        if(self.host == self.HOST_PI):
            self.images_path='/home/pi/WorkSpace/test_images/'

        self.text_file = './lists/captions_' + str(self.total_test_images) + '.txt'
        self.list_file = './lists/images_' + str(self.total_test_images) + '.txt'


    def __getitem__(self):

        return self.x

    def __len__(self):
        return len(self.x)
