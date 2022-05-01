
import tensorflow as tf
print(tf.__version__)
 
from tensorflow.keras.applications import VGG16
 
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,GlobalAveragePooling2D,Dropout,Dense
 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sn
import random
import matplotlib.cm as cm


train_path = './PlantVillage/train'
lists = os.listdir(train_path)
diseases = []
crops = []
file_lst = []
for folder in lists:
    files = os.listdir(os.path.join(train_path,folder))
    files = [folder+'/'+file  for file in files]
    file_lst.extend(files)
    if(folder != 'background'): 
      diseases.extend([folder for i in range(len(files))])
      crops.extend([folder.split(sep='___')[0] for i in range(len(files))])
train_df = pd.DataFrame(list(zip(file_lst,crops,diseases)),columns =["Paths","Crops","Diseases"])

validation_path = './PlantVillage/val'
lists = os.listdir(validation_path)
diseases = []
crops = []
file_lst = []
for folder in lists:
    files = os.listdir(os.path.join(validation_path,folder))
    files = [folder+'/'+file  for file in files]
    file_lst.extend(files)
    if(folder != 'background'): 
      diseases.extend([folder for i in range(len(files))])
      crops.extend([folder.split(sep='___')[0] for i in range(len(files))])
validation_df = pd.DataFrame(list(zip(file_lst,crops,diseases)),columns =["Paths","Crops","Diseases"])



number_images =5

selected_images_df = train_df.sample(number_images)

plt.rcParams["axes.grid"] = False
figure, ax = plt.subplots(1, number_images, figsize=(number_images*7,10)) 
plt.style.use('ggplot')
idx=0
for index, row in selected_images_df.iterrows():
  image = Image.open(os.path.join(train_path,row.Paths))

  disease = row.Diseases.split('___')[1]
  ax[idx].set_title(row.Crops + " : " + disease)
  ax[idx].imshow(image)
  idx= idx + 1

batch_size = 32
image_size = (224,224)

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                  train_path,
                  target_size=image_size,
                  batch_size=batch_size
                  )

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = train_datagen.flow_from_directory(
                  validation_path,
                  target_size=image_size,
                  batch_size=2*batch_size,
                  shuffle=False,
                  )

class_number = train_generator.num_classes

base_model = VGG16(include_top =False,input_shape = (224,224,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(class_number, activation='softmax')(x)
model = Model(base_model.input, predictions)

print(model.summary())

plot_model(model, show_shapes=True, show_layer_names=False)


model.compile(optimizer=optimizers.SGD(learning_rate=1e-3, momentum=0.9), 
              loss='categorical_crossentropy' ,
              metrics = ['accuracy']
              )

nbr_epochs = 10 
history=model.fit(train_generator,
          epochs=nbr_epochs,
          validation_data=validation_generator
)


target_dir = './results/'
model.save(os.path.join(target_dir+'/my_model.h5'))

history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(target_dir+'/history.csv'),index=False)


