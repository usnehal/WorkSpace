#!/usr/bin/env python
# coding: utf-8

# # EYE FOR BLIND
# This notebook will be used to prepare the capstone project 'Eye for Blind'

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:


#Import all the required libraries
import time
import re
import glob
from time import sleep
import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import random
from collections import Counter
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import socket
import pickle


# Let's read the dataset

# ## Data understanding
# 1.Import the dataset and read image & captions into two seperate variables
# 
# 2.Visualise both the images & text present in the dataset
# 
# 3.Create word-to-index and index-to-word mappings.
# 
# 4.Create a dataframe which summarizes the image, path & captions as a dataframe
# 
# 5.Visualise the top 30 occuring words in the captions
# 
# 6.Create a list which contains all the captions & path
# 

# In[3]:


in_nimble = False
in_WSL = False
in_tpu = False

host = socket.gethostname()
if('cuda' in host):
    in_nimble = True
    print("In NimbleBox")
if(host == 'LTsuphale-NC2JM'):
    in_WSL = True
    print("In WSL")


# In[4]:


#Import the dataset and read the image into a seperate variable
# images_path='Images'
# text_file = './captions.txt'
# images_path='/content/drive/MyDrive/TestImages'
# text_file = '/content/drive/MyDrive/captions_small.txt'
# images_path='/content/drive/MyDrive/Images/Images'
# text_file = '/content/drive/MyDrive/captions.txt'
# images_path='./Flickr8K/Images'
# text_file = './Flickr8K/captions.txt'
# images_path='TestImages'
# text_file = './captions_small.txt'
images_path = ''
text_file = ''
total_test_images = 8

if(in_WSL == True):
    images_path='/home/suphale/snehal_bucket/coco/raw-data/train2017/'
if(in_nimble == True):
    images_path='/mnt/disks/user/project/WorkSpace/test_images/'

text_file = './lists/captions_' + str(total_test_images) + '.txt'
list_file = './lists/images_' + str(total_test_images) + '.txt'
# Find out total number of images in the images folder
# all_imgs = glob.glob(images_path + '/*.jpg',recursive=True)
# all_imgs = sorted(all_imgs)
# total_num_images = len(all_imgs)
# print("The total images present in the images folder: {}".format(total_num_images))


# In[5]:


all_imgs = [images_path + line.rstrip() for line in open(list_file)]
all_imgs = sorted(all_imgs)
total_num_images = len(all_imgs)
print("The total images present in the dataset: {}".format(total_num_images))


# In[6]:


#Visualise both the images & text present in the dataset
print("The total images present in the dataset: {}".format(total_num_images))
num_lines = sum(1 for line in open(text_file))
print("The total number of lines in the caption file: {}".format(num_lines))


# In[7]:


# define a function to clean text data
def extract_jpg_caption(line):
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


# In[8]:


#store all the image id here
all_img_id= [] 
#store all the image path here
all_img_vector=[]
#store all the captions here
annotations= [] 
# list of all captions in word list format
annotations_word_list = []


# In[9]:


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
            word_list = caption.split()
            annotations_word_list.append(word_list)
            text += " " + caption
            count += 1
    file.close()
    return text


# In[10]:


#Import the dataset and read the text file into a seperate variable
doc = load_doc(text_file)
print(doc[:300])


# In[11]:


a = set()
for x in annotations_word_list:
    for y in x:
        a.add(y)
print(len(a))


# In[12]:


annotations[:2]


# Randomly pick an image from the data and show its corresponding caption
# 

# In[13]:


random_num = random.randint(0,total_num_images-1)
print(annotations[random_num])
img_path = all_img_vector[random_num]
image = io.imread(img_path)
plt.imshow(image)


# In[14]:


print("Total captions present in the dataset: "+ str(len(annotations)))
print("Total images present in the dataset: " + str(len(all_img_vector)))
total_images = len(all_img_vector)


# Create a dataframe which summarizes the image, path & captions as a dataframe
# 
# Each image id has 5 captions associated with it therefore the total dataset should have 40455 samples.

# In[15]:


df = pd.DataFrame(list(zip(all_img_id, all_img_vector,annotations)),columns =['ID','Path', 'Captions']) 
    


# In[16]:


# Visualise the top 30 occuring words in the captions
vocabulary = [word for word in doc.split()] #write your code here
val_count = Counter(vocabulary)

str_list = ""
for x,y in val_count.most_common(32):
    if(x!='<start>') and (x!='<end>'):
        str_list += " " + x
wordcloud = WordCloud().generate(str_list)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[17]:


print("Total captions present in the dataset: "+ str(len(annotations)))
print("Total images present in the dataset: " + str(len(all_img_vector)))


# ## Pre-Processing the captions
# 1.Create the tokenized vectors by tokenizing the captions fore ex :split them using spaces & other filters. 
# This gives us a vocabulary of all of the unique words in the data. Keep the total vocaublary to top 5,000 words for saving memory.
# 
# 2.Replace all other words with the unknown token "UNK" .
# 
# 3.Create word-to-index and index-to-word mappings.
# 
# 4.Pad all sequences to be the same length as the longest one.

# In[18]:


# create the tokenizer

# your code here
max_tokenized_words = 5000

# tokenizer = Tokenizer(num_words=max_tokenized_words+1,oov_token='<unknown>')
# tokenizer.fit_on_texts(annotations_word_list)

with open('saved_model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# In[19]:


# Create word-to-index and index-to-word mappings.
VOCABULARY_SIZE = len(tokenizer.word_index) + 1
print('Vocabulary Size: {}'.format(VOCABULARY_SIZE))

# print("index of dog = %d" % (tokenizer.word_index['dog']))
# print("word corresponding to index 10 = %s" % (tokenizer.index_word[10]))


# In[20]:


# Create a word count of your tokenizer to visulize the Top 30 occuring words after text processing
data = tokenizer.word_counts


# In[21]:


# Pad each vector to the max_length of the captions ^ store it to a vairable

MAX_SEQ_LENGTH = 25

cap_vector = annotations_word_list
Y_encoded = tokenizer.texts_to_sequences(cap_vector)
cap_vector_encoded_padded = pad_sequences(Y_encoded, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")


# In[22]:


#write your code here
batch_size = 32
def read_image(image_path,label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = tf.cast(image, tf.float32)
    image /= 127.5
    image -= 1.
    return image, label



# In[23]:


from sklearn.model_selection import train_test_split

# split the data into train, validation and final test sets
img_train, final_img_test, cap_train, final_cap_test = train_test_split(all_img_vector, cap_vector_encoded_padded, test_size=0.01, random_state=42)
img_train, img_test, cap_train, cap_test = train_test_split(img_train, cap_train, test_size=0.2, random_state=42)

total_training_images = len(img_train)
total_test_images = len(img_test)
total_final_testing_images = len(final_img_test)
print("Total Images for training=%d" % (total_training_images))
print("Total Images for testing (validation)=%d" % (total_test_images))
print("Total Images for final random testing=%d" % (total_final_testing_images))


# In[24]:


train_dataset = tf.data.Dataset.from_tensor_slices((img_train, cap_train))
test_dataset = tf.data.Dataset.from_tensor_slices((img_test, cap_test))
final_test_dataset = tf.data.Dataset.from_tensor_slices((final_img_test, final_cap_test))



# In[26]:


#write your code here
train_dataset = train_dataset.map(read_image)
train_dataset.shuffle(buffer_size=1024,reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_dataset = test_dataset.map(read_image)
test_dataset.shuffle(buffer_size=1024,reshuffle_each_iteration=True)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# A utility function to display image retrieved from a dataset and its corresponding caption

# In[27]:


# A utility function to display image retrieved from a dataset and its corresponding caption
def show_image_caption_from_dataset(image,label):
    plt.imshow(image)
    for x in label.numpy():
        if(x != 0):
            print(tokenizer.index_word[x], end =" ")


# In[30]:

#Reading the model from JSON file
with open('saved_model/image_features_extract_model.json', 'r') as json_file:
    json_savedModel= json_file.read()
#load the model architecture 
image_features_extract_model = tf.keras.models.model_from_json(json_savedModel)
image_features_extract_model.load_weights('saved_model/image_features_extract_model.h5')


# In[31]:


# write your code to extract features from each image in the dataset
def extract_image_features(sample_img_batch):
    features = image_features_extract_model(sample_img_batch)
    features = tf.reshape(features, [sample_img_batch.shape[0],8*8, 2048])
    return features


# In[32]:


sample_img_batch, sample_cap_batch = next(iter(train_dataset))
sample_img_batch = extract_image_features(sample_img_batch)



# ## Model Building
# 1.Set the parameters
# 
# 2.Build the Encoder, Attention model & Decoder

# In[34]:


embedding_dim = 256 
units = 512
vocab_size = max_tokenized_words + 1
train_num_steps = total_training_images //batch_size #len(total train images) // BATCH_SIZE
test_num_steps = total_test_images //batch_size #len(total test images) // BATCH_SIZE


# ### Encoder

# In[35]:


class Encoder(Model):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        # build your Dense layer with relu activation
        self.dense = tf.keras.layers.Dense(embed_dim, activation='relu')
        
    def call(self, features):
        # extract the features from the image shape: (batch, 8*8, embed_dim)
        features = self.dense(features)
        return features    

# ### Attention model

# In[37]:


from tensorflow.keras.activations import tanh
from tensorflow.keras.activations import softmax

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


# ### Decoder

# In[38]:


class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        # iniitalise your Attention model with units
        self.attention = Attention_model(self.units)
        # build your Embedding layer
        self.embed = tf.keras.layers.Embedding(vocab_size, embedding_dim)
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


# In[39]:

encoder=Encoder(embedding_dim)

decoder=Decoder(embedding_dim, units, vocab_size)

features=encoder(sample_img_batch)

hidden = decoder.init_state(batch_size=sample_cap_batch.shape[0])
dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * sample_cap_batch.shape[0], 1)

predictions, hidden_out, attention_weights= decoder(dec_input, features, hidden)


# In[48]:

decoder.load_weights("saved_model/decoder.h5")
encoder.load_weights("saved_model/encoder.h5")

# In[ ]:


def evaluate(image):
    attention_features_shape = 64
    attention_plot = np.zeros((MAX_SEQ_LENGTH, attention_features_shape))

    hidden = decoder.init_state(batch_size=1)

    # process the input image to desired format before extracting features
    temp_input = tf.expand_dims(read_image(image,[])[0], 0) 
    # Extract features using our feature extraction model
    img_tensor_val = extract_image_features(temp_input)
    # img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    # extract the features by passing the input to encoder
    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(MAX_SEQ_LENGTH):
        # get the output from decoder
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        #extract the predicted id(embedded value) which carries the max value
        predicted_id = tf.argmax(tf.transpose(predictions))
        predicted_id = predicted_id.numpy()[0]
        # map the id to the word from tokenizer and append the value to the result list
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot,predictions

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot,predictions

# In[ ]:


def filt_text(text):
    filt=['<start>','<unk>','<end>'] 
    temp= text.split()
    [temp.remove(j) for k in filt for j in temp if k==j]
    text=' '.join(temp)
    return text

# In[ ]:
import time
rid = np.random.randint(0, total_final_testing_images)
# test_image = final_img_test[rid]
test_image = './413231421_43833a11f5.jpg'

# test_image = './Images/413231421_43833a11f5.jpg'
# test_image = '/content/drive/MyDrive/TestImages/3637013_c675de7705.jpg'

# real_caption = final_cap_test[rid]
# real_caption = '<start> A couple stands close at the water edge <end>'
real_caption = '<start> black dog is digging in the snow <end>'
real_caption = ' '.join([tokenizer.index_word[i] for i in final_cap_test[rid] if i not in [0]])

t0= time.perf_counter()
result, attention_plot,pred_test = evaluate(test_image)
t1 = time.perf_counter() - t0
print("Time elapsed: ", t1)

real_caption=filt_text(real_caption)      

pred_caption=' '.join(result).rsplit(' ', 1)[0]

real_appn = []
real_appn.append(real_caption.split())
reference = real_appn
candidate = pred_caption.split()

score = sentence_bleu(reference, candidate, weights=[1]) #set your weights)
print(f"BLEU score: {score*100}")

print ('Real Caption:', real_caption)
print ('Prediction Caption:', pred_caption)


