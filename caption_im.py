#!/usr/bin/env python
# coding: utf-8

# In[45]:

#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True

import cv2
import matplotlib.pyplot as plt
import re
import collections
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.optimizers import Adam
from keras.layers import *
from keras.models import Model

from tensorflow.keras.models import load_model


# In[46]:


import numpy as np
from keras.preprocessing import image
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input,Dropout,Dense



# In[47]:


model = ResNet50(weights='imagenet',input_shape=(224,224,3))
model_new = Model(model.input,model.layers[-2].output)
#model_new._make_predict_function()


# In[48]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0) # changes (224,224,3)-->(1,224,224,3)
    #Normalization
    img = preprocess_input(img)
    return img


# In[49]:


def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector


# In[50]:


final_model = load_model('./model_weights/model_9.h5')
#final_model._make_predict_function()

# In[51]:



import pickle
with open(".\model_weights\word_to_idx.pkl","rb") as w2i:
    word_to_idx = pickle.load(w2i)
with open(".\model_weights\idx_to_word.pkl","rb") as i2w:
    idx_to_word = pickle.load(i2w)


# In[52]:


max_len = 35
def predict_captions(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = final_model.predict([photo,sequence])
        ypred = ypred.argmax() #Word with max probability (We can also add some randomness)
        word = idx_to_word[ypred]
        in_text += ' '+word
        
        if word == 'endseq':
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


# In[53]:


def predict_caption(photo):
    encoded_image = encode_image(photo)
    photo_2048 = encoded_image.reshape((1,2048))
    #img = plt.imread(photo)
    caption = predict_captions(photo_2048)
    return caption
    #print(caption)
    #plt.imshow(img)
    #plt.axis('off')
    #plt.show()


# In[54]:


caption = predict_caption('he.jpg')


# In[55]:


print(caption)


# In[ ]:




