#!/bin/python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc


### make prediction given a model 

#testname 
timestamp='all-in-one'


def loadModel(jsonStr, weightStr):
    json_file = open(jsonStr, 'r')
    loaded_nnet = json_file.read()
    json_file.close()

    serve_model = tf.keras.models.model_from_json(loaded_nnet)
    serve_model.load_weights(weightStr)
    return serve_model


###to view result, loop to create a heatmap of the same size
def view_result_heatmap(tif,model):
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
      print(j,'out of',height)
      #print(np.squeeze(prob).shape)
      prediction[:,j]=np.squeeze(prob)
      
    prediction_score = prediction
    prediction[prediction>0.5]=1
    prediction[prediction<0.6]=0
    return prediction,prediction_score

def view_result_auc(test_dir, model):
  good = test_dir+'/good/'
  bad = test_dir+'/bad/'
  gt = np.array(([1]*len(os.listdir(good)))+([0]*len(os.listdir(bad))))
  print('len gt ='+str(len(gt)))
  prediction = np.empty([0, 1]) 
 
  count = 0
  batch_input = []
  for test in [good, bad]:
    for f in os.listdir(test):
      img = mpimg.imread(test+f)
      batch_input.append(img/255.0)
      count += 1
      if count%1000 == 0 or f == os.listdir(bad)[-1]:
        prob = model.predict([batch_input])
        batch_input = []
        count = 0 
        prediction = np.append(prediction, prob)
    print(prediction)
  
  prediction_score = prediction
  prediction[prediction>0.5]=1
  prediction[prediction<0.6]=0
  return prediction, prediction_score, gt

def plot_auc(y_true, y_score,imgname ):
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
    plt.savefig(imgname)


test_tif = ['023', '059', '078', '101']
tif = {}
#for idx in test_tif:
#  tif[idx] = TIF(idx)
#  tif[idx].gen_test(save=True, size=1000)

pas = 'all-in-one-val'
base='/root/test2'
test_dir = '/content/test'

i = 0
lrs = [0.001, 0.0001, 0.00001]
decays = [1e-5, 1e-6, 1e-7]
for lr in lrs:
  for decay in decays:
    load_model_name=base+'/all-in-one-val'+'/model-epoch-'+str(i)+'-'+str(lr)+'-'+str(decay)
    model = loadModel(load_model_name+'.json', load_model_name+'.h5')
    sgd = optimizers.SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

# AUC
    prediction, prediction_score, gt = view_result_auc(test_dir, model)

    y_pred = (np.transpose(prediction))
    print(y_pred.shape)
    print(y_pred)
    y_true = gt
    print(y_true.shape)
    y_score = (np.transpose(prediction_score))

    cm = confusion_matrix(y_true, y_pred)
    np.save('cm-fin-'+str(pas)+'-epoch-'+str(i)+'-'+str(lr)+'-'+str(decay), cm)
    plot_auc(y_true,y_score,'auc-fin-'+str(pas)+'-epoch-'+str(i)+'-'+str(lr)+'-'+str(decay)+'.png')

# heatmap
heat_prediction, heat_prediction_score = view_result_heatmap(TIF('078'), model)
plt.imsave('heatmap_prediction-fin-'+str(pas)+'.png',1-np.transpose(heat_prediction))

print('done')


