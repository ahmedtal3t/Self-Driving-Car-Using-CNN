#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D,Dense,MaxPooling2D,Dropout,Flatten
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import ntpath


# In[4]:


datadir = "D:/data-master"


# In[5]:


columns = ['center','left','right','steering','throttle','reverse','speed']


# In[8]:


dataset = pd.read_csv(os.path.join("D:/data-master/driving_log.csv"),names=columns)


# In[9]:


os.path.join("D:/data-master/driving_log.csv")


# In[10]:


dataset.head()


# In[11]:


#Extract the file path
def removePath(path):
  base,tail = ntpath.split(path)
  return tail


# In[12]:


#Transfare the file paths to file names
dataset['center'] = dataset['center'].apply(removePath)


# In[13]:


#Transfare the file paths to file names
dataset['left'] = dataset['left'].apply(removePath)


# In[14]:


##Transfare the file paths to file names
dataset['right'] = dataset['right'].apply(removePath)


# In[15]:


dataset.head()


# In[16]:


num_bins = 25


# In[17]:


hist,bins = np.histogram(dataset['steering'],num_bins)


# In[18]:


print(hist)
print(bins)


# In[20]:


center = (bins[:-1]+bins[1:])*0.5
center


# In[22]:


center1 = []
for i in range(0,len(bins)-1):
  x = (bins[i] + bins[i+1]) * 0.5
  center1.append(x)
center1


# In[23]:


threshold = 500
plt.figure(figsize=(15,10))
plt.bar(center,hist,width=0.05)
plt.xticks(np.linspace(-1,1,25),rotation=90)
(x1,x2) = (np.min(dataset['steering']),np.max(dataset['steering']))
(y1,y2) = (threshold,threshold)
plt.title('Steering Angles')
plt.plot((x1,x2),(y1,y2))


# In[24]:


remove_list = []
for i in range(num_bins):
  List = []
  for j in range(len(dataset['steering'])):
    if dataset['steering'][j] >= bins[i] and dataset['steering'][j] <= bins[i+1]:
      List.append(j)
  List = shuffle(List)
  List = List[threshold:]
  remove_list.extend(List)


# In[25]:


len(dataset['steering']) 


# In[26]:


len(remove_list)


# In[27]:


dataset.drop(dataset.index[remove_list],inplace=True)


# In[28]:


hist,_ = np.histogram(dataset['steering'],num_bins)


# In[29]:


plt.bar(center,hist,width=0.05)
plt.xticks(np.linspace(-1,1,25),rotation=90)
(x1,x2) = (np.min(dataset['steering']),np.max(dataset['steering']))
(y1,y2) = (threshold,threshold)
plt.title('Steering Angles')
plt.plot((x1,x2),(y1,y2))


# In[30]:


dataset.iloc[1]


# In[32]:


datadir


# In[33]:


def loadImageSteering(datadir,dataset):
  imagePath = []
  steeringPath = []
  for i in range(len(dataset)):
    center = dataset.iloc[i][0]
    steering = float(dataset.iloc[i][3])
    imagePath.append(os.path.join(datadir,center))
    steeringPath.append(steering)
  imagePath = np.asarray(imagePath)
  steeringPath = np.asarray(steeringPath)
  return imagePath,steeringPath


# In[34]:


dataset.iloc[0][0]


# In[35]:


imagePath,steeringPath = loadImageSteering(datadir+'/IMG',dataset)


# In[36]:


imagePath[0]


# In[37]:


len(steeringPath)


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(imagePath,steeringPath,random_state=6,test_size=0.2)


# In[40]:


len(x_train)


# In[41]:


plt.figure(figsize=(15,10))
plt.hist(y_train,bins=num_bins,width=0.05)
plt.xticks(np.linspace(-1,1,25),rotation=45)
plt.title("Training Dataset")
plt.show()


# In[42]:


plt.figure(figsize=(15,10))
plt.hist(y_test,bins=num_bins,width=0.05)
plt.xticks(np.linspace(-1,1,25),rotation=45)
plt.title("Testing Dataset")
plt.show()


# In[43]:


def imagePreprocessing(img):
  img = mpimg.imread(img)
  img = img[60:135,:,:]
  img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img,(3,3),0)
  img = cv2.resize(img,(200,66))
  img = img/255
  return img


# In[44]:


image = imagePath[1]
image = mpimg.imread(image)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(image)
axs[0].grid(False)
axs[0].set_title("Original Image")
axs[1].imshow(imagePreprocessing(imagePath[1]))
axs[1].grid(False)
axs[1].set_title("Precessed Image")
plt.show()


# In[45]:


x_train = np.array(list(map(imagePreprocessing,x_train)))


# In[46]:


x_test = np.array(list(map(imagePreprocessing,x_test)))


# In[47]:


def nvidiaModel():
  model = Sequential()
  model.add(Convolution2D(24,(5,5),strides=(2,2),input_shape=(66,200,3),activation="elu"))
  model.add(Convolution2D(36,(5,5),strides=(2,2),activation="elu"))
  model.add(Convolution2D(48,(5,5),strides=(2,2),activation="elu")) 
  model.add(Convolution2D(64,(3,3),activation="elu"))   
  model.add(Convolution2D(64,(3,3),activation="elu"))
  model.add(Dropout(0.5))
  
  model.add(Flatten())
  
  model.add(Dense(100,activation="elu"))
  model.add(Dropout(0.5))
  
  model.add(Dense(50,activation="elu"))
  model.add(Dropout(0.5))
  
  model.add(Dense(10,activation="elu"))
  model.add(Dropout(0.5))
  
  model.add(Dense(1))
  model.compile(optimizer=Adam(lr=1e-3),loss="mse")
  
  return model


# In[48]:


model = nvidiaModel()


# In[49]:


model.summary()


# In[50]:


h = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=30,batch_size=100,shuffle=1,verbose=1)


# In[51]:


plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])


# In[52]:


model.save('car.h5')


# In[53]:


type('car.h5')


# In[ ]:




