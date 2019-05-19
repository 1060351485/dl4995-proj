
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
from tensorflow.keras.models import Sequential
import pytz
from tensorflow.keras import optimizers
from TIF import TIF 
import os
import time

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


def loadModel(jsonStr, weightStr):
    json_file = open(jsonStr, 'r')
    loaded_nnet = json_file.read()
    json_file.close()

    serve_model = tf.keras.models.model_from_json(loaded_nnet)
    serve_model.load_weights(weightStr)
    #rms = optimizers.RMSprop(lr=0.002, decay=0.9, rho=0.9, epsilon=1)
    rms = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    serve_model.compile(optimizer=rms,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    return serve_model


def saveModel(model, model_name):
	# save model to file
	model_json = model.to_json()
	with open(model_name+'.json', 'w') as json_file:
	    json_file.write(model_json)

	model.save_weights(model_name+'.h5')


def initiate_model(new, load_model_name):
	##initiate model either from saving or from new one pretrained from imagenet 
	if new:
		model = Sequential()
		model.add(InceptionV3(weights='imagenet',include_top=False, input_shape=(299, 299, 3)))
		model.add(Flatten())
		model.add(Dense(64, activation='relu'))
		model.add(Dense(1, activation='sigmoid', name='output'))

		model.summary()
		#rms = optimizers.RMSprop(lr=0.002, decay=0.9, rho=0.9, epsilon=1)
		#rms = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
		#model.compile(optimizer=rms, loss='binary_crossentropy')
		return model 
	else:
		model = loadModel(load_model_name+'.json', load_model_name+'.h5')

		return model 

def train(model, epoch,batch):
    trig = ImageDataGenerator(rescale=1./255)
    vig = ImageDataGenerator(rescale=1./255)
    train_generator = trig.flow_from_directory(batch_size=batch, directory='/content/all/train/', shuffle=True, target_size=(299, 299),class_mode='binary',seed=42)
    val_generator = vig.flow_from_directory(batch_size=batch, directory='/content/all/validation/', target_size=(299, 299),class_mode='binary')
    model.fit_generator(train_generator, validation_data =val_generator,epochs = epoch, steps_per_epoch=None,validation_steps = None)

	

tz = pytz.timezone("America/New_York")

timestamp = 'all-in-one-val' 
os.system('mkdir -p ' + timestamp)
base='/root/test2/'

lrs = [0.001, 0.0001, 0.00001]
decays = [1e-5, 1e-6, 1e-7]
for lr in lrs:
  for decay in decays:
    print('lr=', lr, ', decay=', decay)
    model = initiate_model(new=True, load_model_name='')
    sgd = optimizers.SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    epoch = 1
    batch_size = 32 
    i = 0 
    train(model,epoch,batch_size)
    model_name= base+timestamp+'/model-epoch-'+str(i)+'-'+str(lr)+'-'+str(decay)
    saveModel(model, model_name)
