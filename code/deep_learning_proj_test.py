# -*- coding: utf-8 -*-
"""Deep_Learning_Proj_test (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oe0pej82OkGlw-iHy-b6iKwOF7Q_9zUC

TODO (as of 15 Apr)

1.  Model loss=0, outputting NaN(fixed by /255.0)

2.  Solve problem of duplicated names in model (currently using two different models)

2.  Implement Preprocess (take away blank parts)

3.  Speed issues:
curernt 5000 data points at 64-sample batch, takes 150s per epoch;
per image has on average 200k data points or more, 50 epochs takes 
50x150x(200/5) /3600= 83 hours, 20 images=huge
  
  1.  Better data generator?
  2.  Choice of area in the picture (most blank anyway)
  3.  Larger batchsize? but GPU approaching limit already

4.  Re-balance data set
   1.  Implement different weight loss (probably easier)
   2.  Multiples positive samples
   
 
 Apr 16
 1. add label_generator in TIF class, map(id, i, j) ->label
 2. add read_center in TIF class, read patch centered at x, y
 3. add train/val/test generator in TIF class, without augmentation so far
 4. move find_tissue_pixels to TIF
 5. add crop_show, move crop and plot in this func for test
 
 
 Apr 17
 
 1. train/val/test, to generate test data, call gen_test_data
 2. save/load model
 
* 无：035w(1.5), 059w(1.4), 
* 点：023w(1.6),019w(1.4)，012w(1.5), 005w(1.4), 002w,(1.6) 001w(3), 057w(1.4)，081p(1.1)
* 小团: 016w(1.4), 031w(1.4), 064w(1.5)， 075p(0.9)， 091p(0.5)， 094p(1.5)，096p(1.2)，101p(1.4)
* 大：078p(1.5)，110p(1.4)

* tif_list = ['001', '002', '005', '012', '016', '019', '023', '031', '035', '057', '059', '064', '075', '078', '081', '084', '091', '094', '096', '101', '110']
* test_id = ['059', '078', '101', '023']
* train_val_id = ['001', '002', '005', '012', '016', '019', '031', '035', '057', '064', '081', '084', '091', '094', '096', '110']
* heatmap_id = ['075']

## About

This starter code shows how to read slides and tumor masks from the [CAMELYON16](https://camelyon17.grand-challenge.org/Data/) dataset. It will install and use [OpenSlide](https://openslide.org/), the only non-Python dependency. Note that OpenSlide also includes a [DeepZoom viewer](https://github.com/openslide/openslide-python/tree/master/examples/deepzoom), shown in class. To use that, though, you'll need to install and run OpenSlide locally.

### Training data

This [folder](https://drive.google.com/drive/folders/1rwWL8zU9v0M27BtQKI52bF6bVLW82RL5?usp=sharing) contains a few slides and tumor masks prepared in advance with ASAP. This amount of data (or less!) should be sufficient for your project. The goal is to build a thoughtful end-to-end prototype, not to match the accuracy from the [paper](https://arxiv.org/abs/1703.02442) discussed in class. If you would like more data than has been provided, you will need to use ASAP to convert it into an appropriate format.
"""

# %matplotlib inline
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

"""read data"""

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
    os.system('rm -rf /content/'+self.tif_id)
    for dd in [self.train_dir, self.val_dir]:
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
        os.system("curl -O "+slide_url)

    # Download the tumor mask
    if not os.path.exists(self.tumor_mask_path):
      os.system("curl -O "+mask_url)

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
    
    if(save):
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
    
    if(save):
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

tif_list = ['001', '002', '005', '012', '016', '019', '023', '031', '035', 
            '057', '059', '064', '075', '078', '081', '084', '091', '094',
            '096', '101', '110']
test_id = ['059', '078', '101', '023']
train_val_id = ['001', '002', '005', '012', '016', '019', '031', '035', 
            '057', '064', '081', '084', '091', '094', '096', '110']

heatmap_id = ['075']

tif_id = '091'
tif91 = TIF(tif_id)

# Example: read the entire slide at level 5

# Higher zoom levels may not fit into memory.
# You can use the below function to extract regions from higher zoom levels 
# without having to read the entire image into ram.

# Use the sliding window approach discussed in class to collect training
# data for your classifier. E.g., slide a window across the slide (for
# starters, use a zoomed out view, so you're not working with giant images).
# Save each window to disk as an image. To find the label for that image, 
# check to the tissue mask to see if the same region contains cancerous cells.

# Important: this is tricky to get right. Carefully debug your pipeline before
# training your model. Start with just a single image, and a relatively 
# low zoom level.

# (x, y) tuple giving the top left pixel in the level ***0*** reference frame
# x/w is horizon, y/h is vertical
level_for_read = 5
slide_image = tif91.read_slide(x=0, 
                               y=0,
                               level=level_for_read, 
                               width=tif91.dimi(level_for_read)[0], 
                               height=tif91.dimi(level_for_read)[1])
print(tif91.dimi(level_for_read))
print(slide_image.shape)
#plt.figure(figsize=(10,10), dpi=100)
#plt.imshow(slide_image)

a = tif91.read_center(x=150*128,y=200*128,width=150,height=150,level=5,show=False)

def apply_mask(im, mask, color=(255,0,0)):
    masked = np.copy(im)
    for x,y in mask: masked[x][y] = color
    return masked

# tissue_regions = apply_mask(slide_image, tissue_pixels)
# plt.imshow(tissue_regions)


"""pre-processing"""

good, bad=tif91.gen_data(save=True, balance=True)
print(len(good), len(bad))

batch_size = 32
train_generator, val_generator = tif91.load_dir(batch_size, 299,299)

print(train_generator.class_indices)

####generate training data

# x_train, y_train = tif91.label_generator(level=7)

tif_dict = {}
tif_dict['091']=tif91

# crop test tifs in test list, save to '/content/test'
def gen_test_data(test):
  test_dir = '/content/test'
  test_bad_dir = test_dir + '/bad'
  test_good_dir = test_dir + '/good'
  save = None
  for idx in test:
    tif = TIF(idx)
    tif.gen_test(save=True, balance=True)
    save = tif
  return save.load_test_dir() # test data generator


from tensorflow.keras.models import Sequential
model = Sequential()
model.add(InceptionV3(weights='imagenet',include_top=False, input_shape=(299, 299, 3)))
model.add(Dense(32, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid', name='output'))

model.summary()

from tensorflow.keras import optimizers
sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')

"""train"""

x,y = next(train_generator)
x1 = x[0]
x1 = np.expand_dims(x1,axis=0)

# train_steps = len(x_train[:5000]) // batch_size

# model.fit_generator(train_generator, validation_data= None, 
#                     epochs = 1, steps_per_epoch = train_steps, validation_steps = None)

model.fit_generator(train_generator, validation_data =val_generator,epochs = 10,
                    steps_per_epoch=None,validation_steps = None)

# save model to file
model_name = 'model-091'
model_json = model.to_json()
with open(model_name+'.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights(model_name+'.h5')

load_model_name = 'model-091'
def loadModel(jsonStr, weightStr):
    json_file = open(jsonStr, 'r')
    loaded_nnet = json_file.read()
    json_file.close()

    serve_model = tf.keras.models.model_from_json(loaded_nnet)
    serve_model.load_weights(weightStr)
    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    serve_model.compile(optimizer=sgd,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    return serve_model

model2 = loadModel(load_model_name+'.json', load_model_name+'.h5')

"""test"""



"""view result"""

###to view result, loop to create a heatmap of the same size

def view_result(tif):
    width = tif.dimi(7)[0]
    height = tif.dimi(7)[1]
    prediction = np.zeros((width,height))
    
    for j in range(height):
      #print(j)
      batch_input=[]
      for i in range(width):
        #print(i,j)
        

       
        x1 = tif.read_center(x=i*128,y=j*128,width=299, height=299,level=5)
        batch_input.append(x1/255.0)

#         x1 = np.expand_dims(x1,axis=0)
#         x2 = np.expand_dims(x2,axis=0)


      prob = model.predict([batch_input])
#       print(len(batch_input))
#       print(len(prob))
#       print(i, j, prob)
      print(j)
      #print(np.squeeze(prob).shape)
      prediction[:,j]=np.squeeze(prob)
      
    prediction_score = prediction
    prediction[prediction>0.5]=1
    prediction[prediction<0.6]=0
    return prediction,prediction_score

prediction,prediction_score = view_result(tif91)
print(prediction[350])
#plt.imshow(prediction)

#plt.imshow(1-np.transpose(prediction))

gt = tif91.read_mask(x=0, 
                      y=0, 
                      level=7, 
                      width=tif91.tumor_mask.level_dimensions[7][0], 
                      height=tif91.tumor_mask.level_dimensions[7][1],
                      show=True)

y_pred = (1-np.transpose(prediction)).flatten()
y_true = gt[:,:,0].flatten()
y_score = (1-np.transpose(prediction_score)).flatten()


print(len(y_true))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)

from sklearn.metrics import roc_curve,auc
def plot_auc(y_true, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y_true[:], y_score[:])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
#plot_auc(y_true, y_score)


tif_=TIF('078')

mask = tif_.read_mask(x=0, 
                      y=0, 
                      level=7, 
                      width=tif91.tumor_mask.level_dimensions[7][0], 
                      height=tif91.tumor_mask.level_dimensions[7][1],
                      show=False)

#plt.imshow(mask[:,:,0])

def test_tif(tif):
    pred,score = view_result(tif)
    print('predicted')
    plt.imshow(1-np.transpose(pred))
  
    y_pred = (1-np.transpose(pred)).flatten()
    y_true = gt[:,:,0].flatten()
    y_score = (1-np.transpose(score)).flatten()

    confusion_matrix(y_true, y_pred)

    #plot_auc(y_true, y_score)

test_tif(tif_)

tif_.dimi(7)

tif_.dimi_m(7)

mask = tif_.read_mask(x=0, 
                      y=0, 
                      level=7, 
                      width=tif_.tumor_mask.level_dimensions[7][0], 
                      height=tif_.tumor_mask.level_dimensions[7][1],
                      show=False)

tif_dict =dict()
for idx in tif_list:
  tif_dict[idx]=TIF(idx,mask_only=True)
  a = tif_dict[idx]
  print(idx)
  mask = a.read_mask(x=0, 
                      y=0, 
                      level=7, 
                      width=a.tumor_mask.level_dimensions[7][0], 
                      height=a.tumor_mask.level_dimensions[7][1],
                      show=False)
#  plt.imshow(mask[:,:,0])
#  plt.show()