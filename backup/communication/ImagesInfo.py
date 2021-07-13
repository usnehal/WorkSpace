import Logger
import Util

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

        doc = self.load_doc(self.cfg.text_file)

    def load_doc(self, filename):
        #your code here
        file  = open(filename, 'r') 
        Lines = file.readlines() 
        
        text = ""
        count = 0
        for line in Lines:
            jpg_path, caption = Util.extract_jpg_caption(self.cfg.images_path, line)
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
            real_caption=Util.filt_text(real_caption)
            real_appn.append(real_caption.split())
        return real_appn
