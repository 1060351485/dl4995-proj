#!/bin/py
import matplotlib.pyplot as plt
import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
import os
from PIL import Image
from skimage.color import rgb2gray
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import random
from sklearn.model_selection import train_test_split
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate,Flatten
from tensorflow.keras.models import Model

import pytz
tz = pytz.timezone("America/New_York")

class TIF(object):
  def __init__(self, tif_id, mask_only=False):
    self.tif_id = tif_id
    self.slide_path = 'tumor_'+tif_id+'.tif'
    self.tumor_mask_path = 'tumor_'+tif_id+'_mask.tif'
    self.download(mask_only)
    if not mask_only:
      self.slide = open_slide(self.slide_path)
    self.tumor_mask = open_slide(self.tumor_mask_path)
    self.init_generator()
  
  def init_generator(self):
    self.base_dir='/content/'+self.tif_id+'/'
    self.train_dir=self.base_dir+'train'
    self.val_dir=self.base_dir+'validation'
    self.test_dir='/content/test'
    if(not os.path.exists(self.val_dir+'/bad/') or len(os.listdir(self.val_dir+'/bad/')) == 0):
        os.system('rm -rf /content/'+self.tif_id)
        for dd in [self.train_dir, self.val_dir, self.test_dir]:
            for tt in ['/good', '/bad']:
                os.system('mkdir -p '+dd+ tt)
    self.trig = ImageDataGenerator(rescale=1./255)
    self.vig = ImageDataGenerator(rescale=1./255)
    self.ttig = ImageDataGenerator(rescale=1./255)
  
  def slide(self):
    return self.slide
  
  def get_id(self):
    return self.tif_id
  
  def tumor_mask(self):
    return self.tumor_mask
  
  def dimi(self, i):
    return self.slide.level_dimensions[i]
  
  def dimi_m(self, i):
    return self.tumor_mask.level_dimensions[i]
  
  # Download an example slide and tumor mask
  # Important note: the remainder are in a Google Drive folder, linked above.
  # You will need to host them on your own, either in Google Drive, or by using
  # the cloud provider of your choice.
  def download(self,mask_only):
    slide_url = 'https://storage.googleapis.com/4995dl/%s' % self.slide_path
    mask_url = 'https://storage.googleapis.com/4995dl/%s' % self.tumor_mask_path
    # Download the whole slide image
    if not mask_only:
      if not os.path.exists(self.slide_path):
        os.system('curl -O '+slide_url)

    # Download the tumor mask
    if not os.path.exists(self.tumor_mask_path):
      os.system('curl -O '+mask_url)

  # See https://openslide.org/api/python/#openslide.OpenSlide.read_region
  # Note: x,y coords are with respect to level 0.
  # There is an example below of working with coordinates
  # with respect to a higher zoom level.

  # Read a region from the slide
  # Return a numpy RBG array
  def read_slide(self, x, y, level, width, height, as_float=False, show=False):
    im = self.slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
      im = np.asarray(im, dtype=np.float32)
    else:
      im = np.asarray(im)
    assert im.shape == (height, width, 3)
    if(show):
      plt.imshow(im)
    return im
    
  def read_mask(self, x, y, level, width, height, as_float=False, show=False):
    im = self.tumor_mask.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
      im = np.asarray(im, dtype=np.float32)
    else:
      im = np.asarray(im)
    assert im.shape == (height, width, 3)
    if(show):
      plt.imshow(im[:,:,0])
    return im
  
  # we need data of the following format:
  #input: (k,x,y) where k is the slice index, x, y are the coordinates of the heat map, 
  #output: 1/0 of the heatmap value

  
  #
  # re-map (i, j) -> label
  # 1*1 cell in level 7 => 4*4 cell in level 5
  # if 4*4 cell in level 5 tumor <= threshold, re-label it to "not tumor"
  # 
  def label_generator(self, level=5, label_level=7, threshold=0):
    
    print("label_generator called, level=", level)
    start = datetime.datetime.now(tz)
    print("start time", start)
    
    dimi7 = self.dimi(label_level)
    heatmap = self.read_mask(x=0, 
                             y=0,
                             level=level, 
                             width=self.dimi_m(level)[0], 
                             height=self.dimi_m(level)[1])
    
    heatmap = heatmap[:,:,0]
    print(heatmap.shape)
    stride = 2**(label_level-level)
    
    x=[]
    y=[]
    for i in range(dimi7[0]):
      for j in range(dimi7[1]):
        id= self.get_id()
        x.append((i,j))
        cnt = 0
        for m in range(stride):
          for n in range(stride):
            a = i*2**(label_level-level)+m
            b = j*2**(label_level-level)+n
            # !!! mask.shape = (420, 480, 3)
            # !!! self.dimi_m(level) = (480, 420)
            # !!! so use heatmap[b][a]
            if(a < self.dimi_m(level)[1] and b < self.dimi_m(level)[0]):
              if(heatmap[b][a] == 1):
                cnt+=1
        if(cnt <= threshold):
          y.append(0)
        else:
          y.append(1)
      if (i % 10 == 0):
        print("fin line", i)
    
    end = datetime.datetime.now(tz)
    print("start time", end)
    print("cost", end-start)
    
    return x,y
  
  # x, y is center of cropped image(level0)
  # width, height is at "level" level
  def read_center(self, x, y, width, height, level, show=False, k=None,):
    #this would be used for data pipeline
    #what it does:it creates an tif object(if not already available),then extract 
    #image from the index given from the heatmap(level 7)
    #at the moment we are giving it two image of different scale

    #zoom at level
    i = int(x-0.5*width*(2**(level)))
    j = int(y-0.5*height*(2**(level)))

    zoom = self.read_slide(x=i, 
                          y=j,
                          level=level, 
                          width=width, 
                          height=height)
    if(show):
      plt.imshow(zoom)
    return zoom
  
  # 
  # input x, y is top left corner of 4*4 detection area
  # x, y is at level 0
  # output patch is 299*299 at level "level" which is 5 by default
  # stride 128 is at level 0, is 4 at level 5
  # width and height are at level 5
  # 
  def read_zoom(self, x, y, width=299, height=299, level=5, show=False, k=None):
    
    stride = 2**(7-level)
    
    i = int(x+(stride-width)*0.5*(2**level))
    j = int(y+(stride-height)*0.5*(2**level))
#     print(i, j)
    zoom = self.read_slide(x=i,
                           y=j,
                           level=level, 
                           width=width, 
                           height=height)
    if(show):
      plt.imshow(zoom)
    return zoom
  
  
  # width and height is at level "level"=5
  # 
  
  def gen_data(self, width=299, height=299, level=5, save=False, 
               balance=False, tumors=0, intensity=0.8, 
               tissue_percent=0.5):
    
    width = math.floor(width/(2**level))
    height = math.floor(height/(2**level))
    stride = 2**(7-level)
    
    x_train, y_train = self.label_generator(level=7, threshold=tumors)
    
    good = [] # none tumor(not gray or black) coordinates
    bad = [] # tumor coordinates
    idx = 0
    start = datetime.datetime.now(tz)
    print("start time", start)
    # Read in each input, perform preprocessing and get labels
    for path in range(len(x_train)):
        i,j = x_train[path]
        y = y_train[path]

        detect = self.read_zoom(i*128, j*128, stride, stride, level)
        
        # check percent_tissue of this patch
        # discard if < tissue_percent
        im_gray = rgb2gray(detect)
        assert im_gray.shape == (detect.shape[0], detect.shape[1])
#         plt.imshow(detect)
        
        # im_gray <= intensity and not all black
        indices = np.where(np.logical_and(im_gray <= intensity, im_gray != 0))
        rslt = list(zip(indices[0], indices[1]))
        
        percent_tissue = len(rslt) / float(detect.shape[0] * detect.shape[0]) * 100
#         print ("%d tissue_pixels pixels (%.1f percent of the image)" % (len(rslt), percent_tissue)) 
        if(percent_tissue < tissue_percent):
          continue
        
        if(y == 1):
          bad.append((i, j))
        else:
          good.append((i, j))
        idx+=1
        if(idx %1000 == 0):
          print("%d patches cropped, i=%d, j=%d"% (idx, i, j))
    
    # print time cost
    end = datetime.datetime.now(tz)
    print("end time", end)
    print("cost", end-start)
    
    # balance tumor/none patches
    if(balance):
      if(len(bad) <= len(good)):
        good = random.choices(good, k=len(bad))        
    
    if(save and len(os.listdir(self.val_dir+'/bad/'))==0 ):
      X_train_g, X_val_g, y_train_g, y_val_g = train_test_split(good, [0]*len(good), test_size=0.25, random_state=1)

      for x, y in X_train_g:
        im_m = self.read_zoom(x*128, y*128)
        im = Image.fromarray(im_m)
        im.save(self.train_dir+'/good/'+self.tif_id+'_'+str(x)+'_'+str(y)+'.jpeg','JPEG')

      for x, y in X_val_g:
        im_m = self.read_zoom(x*128, y*128)
        im = Image.fromarray(im_m)
        im.save(self.val_dir+'/good/'+self.tif_id+'_'+str(x)+'_'+str(y)+'.jpeg','JPEG')

        
      X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(bad, [0]*len(bad), test_size=0.25, random_state=1)

      for x, y in X_train_b:
        im_m = self.read_zoom(x*128, y*128)
        im = Image.fromarray(im_m)
        im.save(self.train_dir+'/bad/'+self.tif_id+'_'+str(x)+'_'+str(y)+'.jpeg','JPEG')

      for x, y in X_val_b:
        im_m = self.read_zoom(x*128, y*128)
        im = Image.fromarray(im_m)
        im.save(self.val_dir+'/bad/'+self.tif_id+'_'+str(x)+'_'+str(y)+'.jpeg','JPEG')

    # Return a tuple of (input,output) to feed the network
    
    return np.array(good), np.array(bad)
    
  def gen_test(self, width=299, height=299, level=5, save=False, 
               size=1000, tumors=0, intensity=0.8, 
               tissue_percent=0.5):
    
    width = math.floor(width/(2**level))
    height = math.floor(height/(2**level))
    stride = 2**(7-level)
    
    x_train, y_train = self.label_generator(level=7, threshold=tumors)
    
    good = [] # none tumor(not gray or black) coordinates
    bad = [] # tumor coordinates
    idx = 0
    start = datetime.datetime.now(tz)
    print("start time", start)
    # Read in each input, perform preprocessing and get labels
    for path in range(len(x_train)):
        i,j = x_train[path]
        y = y_train[path]

        detect = self.read_zoom(i*128, j*128, stride, stride, level)
        
        # check percent_tissue of this patch
        # discard if < tissue_percent
        im_gray = rgb2gray(detect)
        assert im_gray.shape == (detect.shape[0], detect.shape[1])
#         plt.imshow(detect)
        
        # im_gray <= intensity and not all black
        indices = np.where(np.logical_and(im_gray <= intensity, im_gray != 0))
        rslt = list(zip(indices[0], indices[1]))
        
        percent_tissue = len(rslt) / float(detect.shape[0] * detect.shape[0]) * 100
#         print ("%d tissue_pixels pixels (%.1f percent of the image)" % (len(rslt), percent_tissue)) 
        if(percent_tissue < tissue_percent):
          continue
        
        if(y == 1):
          bad.append((i, j))
        else:
          good.append((i, j))
        idx+=1
        if(idx %1000 == 0):
          print("%d patches cropped, i=%d, j=%d"% (idx, i, j))
        
    # print time cost
    end = datetime.datetime.now(tz)
    print("end time", end)
    print("cost", end-start)
    
    if(len(bad) <= size/2):
      good = random.choices(good, k=size-len(bad))
    else:
      bad = random.choices(bad, k=size//2)
      good = random.choices(good, k=size//2)        
    
    if save:
      for x, y in good:
        im_m = self.read_zoom(x*128, y*128)
        im = Image.fromarray(im_m)
        im.save(self.test_dir+'/good/'+self.tif_id+'_'+str(x)+'_'+str(y)+'.jpeg','JPEG')
      for x, y in bad:
        im_m = self.read_zoom(x*128, y*128)
        im = Image.fromarray(im_m)
        im.save(self.test_dir+'/bad/'+self.tif_id+'_'+str(x)+'_'+str(y)+'.jpeg','JPEG')
    # Return a tuple of (input,output) to feed the network
    return np.array(good), np.array(bad)

  #
  # load data to image generator, after call gen_data and save to file
  #
  def load_dir(self, batch, width, height):
    train_data_gen = self.trig.flow_from_directory(batch_size=batch, 
                                                   directory=self.train_dir, 
                                                   shuffle=True, # Best practice: shuffle the training data
                                                   target_size=(width, height),
                                                   class_mode='binary',
                                                   seed=42)
    val_data_gen = self.vig.flow_from_directory(batch_size=batch, 
                                                directory=self.val_dir, 
                                                target_size=(width, height),
                                                class_mode='binary')
    return train_data_gen, val_data_gen
  
 
  def load_test_dir(self, batch, width, height):
    test_data_gen = self.ttig.flow_from_directory(batch_size=batch, 
                                                  directory=self.test_dir, 
                                                  target_size=(width, height),
                                                  class_mode='binary')
    return test_data_gen
 

  def info(self):
    print ("Read WSI from %s with width: %d, height: %d" % (self.slide_path, 
                                                            self.slide.level_dimensions[0][0], 
                                                            self.slide.level_dimensions[0][1]))
    print ("Read tumor mask from %s" % (self.tumor_mask_path))

    print("Slide includes %d levels", len(self.slide.level_dimensions))
    for i in range(len(self.slide.level_dimensions)):
        print("Level %d, dimensions: %s downsample factor %d" % (i, 
                                                                 self.slide.level_dimensions[i], 
                                                                 self.slide.level_downsamples[i]))
        print('mask levels: %d, slide levels: %d' % (len(self.tumor_mask.level_dimensions), len(self.slide.level_dimensions)))
        #no need to check levels as may not be same
        assert len(self.tumor_mask.level_dimensions)>7
        assert len(self.slide.level_dimensions)>7
        #assert self.tumor_mask.level_dimensions[i][0] == self.slide.level_dimensions[i][0]
        #assert self.tumor_mask.level_dimensions[i][1] == self.slide.level_dimensions[i][1]

    # Verify downsampling works as expected
    width, height = self.slide.level_dimensions[7]
    assert width * self.slide.level_downsamples[7] == self.slide.level_dimensions[0][0]
    assert height * self.slide.level_downsamples[7] == self.slide.level_dimensions[0][1]  
    
  
  # As mentioned in class, we can improve efficiency by ignoring non-tissue areas 
  # of the slide. We'll find these by looking for all gray regions.
  def find_tissue_pixels(self, level=5, intensity=0.8, show=False):
    image = tif91.read_slide(x=0, 
                             y=0,
                             level=level, 
                             width=tif91.dimi(level)[0], 
                             height=tif91.dimi(level)[1])
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
#     plt.imshow(im_gray)
    indices = np.where(im_gray <= intensity)
    
    rslt = list(zip(indices[0], indices[1]))
    
    if(show):
      percent_tissue = len(rslt) / float(image.shape[0] * image.shape[0]) * 100
      print ("%d tissue_pixels pixels (%.1f percent of the image)" % (len(rslt), percent_tissue)) 
    
    return rslt
  
  
  #
  # crop and show for test
  #
  def crop_show(self, width, height, level):
    level0_dim = self.dimi(0)
    print(self.dimi(level))
    w = width
    h = height
    i = 0
    j = 0
    level = level
    idx = 1

    plt.figure(figsize=(10,10), dpi=100)
    while j < level0_dim[1]:
      while i < level0_dim[0]:
#         w = width
#         if(i + (2**level)*w > level0_dim[0]):
#           w = (level0_dim[0]-i)/(2**level)+1
        slide_piece = self.read_center(x=i, 
                                     y=j,
                                     level=level, 
                                     width=w, 
                                     height=h)

        plt.subplot(math.ceil(tif91.dimi(level)[1]*1.0/h), math.ceil(tif91.dimi(level)[0]*1.0/(w)), idx)

        plt.imshow(slide_piece)
        idx += 1
        i += 2**level*w
      i = 0
      j += 2**level*h

    plt.show()
