from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
import os
import sys



def get_data(path,limit):
  
  ''' Given a path, this function returns four lists created from the path -
  
     eg. 'path = 'train_path'' returns data in the NORMAL / PNEUMONIA folders
     corresponding to this path
     
     limit is imposed to save memory, this means the no. of data obtained is only
     up to that limit
  
  
     samples_normal : all samples for the 'NORMAL' images
     filenames_normal : filenames corresponding to samples_normal
      
     samples_pneumonia : all samples for the 'PNEUMONIA' images
     filenames_pneumonia: filenames corresponding to samples_pneumonia
      
  '''

  samples_normal, samples_pneumonia = [], []
  filenames_normal, filenames_pneumonia = [], []

  # Get data for normal images
  for filename in os.listdir(path + 'NORMAL')[:limit]:
  
    if filename != '.DS_Store':
      image = Image.open(os.path.join(path + 'NORMAL', filename))
      imarray = np.array(image)
      samples_normal.append(imarray)
      filenames_normal.append(filename)
  
  print('Files in first {} images of {} appended!'.format(limit,path + 'NORMAL'))

  
  # Get data for pneumonia images
  for filename in os.listdir(path + 'PNEUMONIA')[:limit]:
  
    if filename != '.DS_Store':
      image = Image.open(os.path.join(path + 'PNEUMONIA', filename))
      imarray = np.array(image)
      samples_pneumonia.append(imarray)
      filenames_pneumonia.append(filename)
  
  print('Files in first {} images of {} appended!'.format(limit,path + 'PNEUMONIA'))


  return samples_normal, samples_pneumonia, filenames_normal, filenames_pneumonia
 

def visualise_images(samples_normal, samples_pneumonia):
  
  plt.figure(figsize=(15,15))
  all_samples = samples_normal + samples_pneumonia

  # Plot 6 random samples
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.subplots_adjust(bottom=0.3, top=0.9, hspace=0)
    k = np.random.choice(len(all_samples))
    label = 'NORMAL' if k <= (0.5*len(all_samples)-1) else 'PNEUMONIA'
    plt.xlabel('True label : {}'.format(label),fontsize=15)
    plt.imshow(all_samples[k].squeeze(), cmap=plt.get_cmap('gray'), interpolation='nearest')


# Create train and test data gen
def create_train_test_val_generator(train_path,test_path,val_path,img_size=(96,96),batch_size=16,
                                    shear_range=0.2,zoom_range=0.2):
  
  # Create generators
  train_datagen = ImageDataGenerator(
      rescale=1./255,
      shear_range=shear_range,
      zoom_range=zoom_range,
      #width_shift_range=0.2,
      #height_shift_range=0.2,
      horizontal_flip=True)

  test_datagen = ImageDataGenerator(rescale=1./255)
  
  # Generate from path
  train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')
  
  validation_generator = test_datagen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')
  
  test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')
  

  return train_generator, validation_generator, test_generator