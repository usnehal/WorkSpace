import json
import random
import os
import shutil
import socket
from tqdm import tqdm
import pandas as pd

import communication.Logger as Logger

in_nimble = False
in_WSL = False
in_tpu = False
in_pi = False

host = socket.gethostname()
if('cuda' in host):
    in_nimble = True
    print("In NimbleBox")
if(host == 'LTsuphale-NC2JM'):
    in_WSL = True
    print("In WSL")    
if(host == 'raspberrypi'):
    in_pi = True
    print("In raspberry-pi")    

if(in_WSL == True):
    coco_dir = '/home/suphale/snehal_bucket/coco/raw-data'
if(in_nimble == True):
    coco_dir = '/mnt/disks/user/project/coco'
if(in_nimble == True):
    coco_dir = '/mnt/disks/user/project/coco'

test_image_dir = './test_images'
annotation_dir = coco_dir + '/annotations'
train_2017_dir = coco_dir + '/train2017'
captions_val2017_json = annotation_dir + '/captions_val2017.json'
captions_train2017_json = annotation_dir + '/captions_train2017.json'
instances_train2017_json = annotation_dir + '/instances_train2017.json'

Logger.milestone_print('reading json %s' % (instances_train2017_json))
f = open(instances_train2017_json,)
data_instances = json.load(f)

Logger.milestone_print('reading json %s' % (captions_train2017_json))
f = open(captions_train2017_json,)
data_captions = json.load(f)

shutil.rmtree(test_image_dir,ignore_errors = True)
os.mkdir(test_image_dir)

# total_test_images = 118287
# test_images_list = [10,100,1000, 10000, 118287]
test_images_list = [10,100,1000]
for total_test_images in test_images_list:
    Logger.milestone_print("total_test_images=%d" % (total_test_images))
    copy_required = False
    if(total_test_images == 10) or (total_test_images == 100):
        copy_required = True
    
    test_image_captions_txt = './lists/captions_' + str(total_test_images) + '.txt'
    test_image_list_txt = './lists/images_' + str(total_test_images) + '.txt'
    test_instances_list_txt = './lists/instances_' + str(total_test_images) + '.txt'

    df_captions_csv = './lists/df_captions_' + str(total_test_images) + '.csv'
    df_instances_csv = './lists/df_instances_' + str(total_test_images) + '.csv'

    captions_file = open(test_image_captions_txt, "w")
    image_list_file = open(test_image_list_txt, "w")
    instances_list_file = open(test_instances_list_txt, "w")

    total_images = len(data_captions['images'])
    print("total_images = %d" % (total_images))
    print("total_test_images = %d" % (total_test_images))
    df_caption = pd.DataFrame(columns = ['FileName' , 'Caption']) 
    df_annotation = pd.DataFrame(columns = ['FileName' , 'Caption']) 

    if (total_images < total_test_images):
        total_test_images = total_images
    print("total images = %d total test images = %d" % (total_images,total_test_images))
    random_list = random.choices(data_captions['images'],k=total_test_images)
    for x in tqdm(random_list):
        file_name= x['file_name']
        image_id = x['id']
        image_list_file.write(file_name+'\n')
        if(copy_required == True):
            shutil.copyfile(train_2017_dir + '/' + file_name, test_image_dir + '/' + file_name)
        for a in data_captions['annotations']:
            if(a['image_id'] == image_id):
                captions_file.write(file_name + ',' + a['caption']+'\n')
                series_obj = pd.Series( [file_name, a['caption']], index=df_caption.columns )
                df_caption = df_caption.append( series_obj, ignore_index=True)
        for a in data_instances['annotations']:
            if(a['image_id'] == image_id):
                instances_list_file.write(file_name + ',' + str(a['category_id'])+'\n')
                series_obj = pd.Series( [file_name, a['category_id']], index=df_annotation.columns )
                df_annotation = df_annotation.append( series_obj, ignore_index=True)

    captions_file.close()
    image_list_file.close()
    instances_list_file.close()
    df_caption.to_csv(df_captions_csv)
    df_annotation.to_csv(df_instances_csv)

df_categories = pd.DataFrame(columns = ['id' , 'name'])
for x in data_instances['categories']:
    series_obj = pd.Series( [x['id'], x['name']], index=df_categories.columns )
    df_categories = df_categories.append( series_obj, ignore_index=True)
df_categories.to_csv('categories.csv')