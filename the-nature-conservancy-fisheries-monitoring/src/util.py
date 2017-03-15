from os import path
import tensorflow as tf
import numpy as np
import re
import config

# If you want this to work do not move this file
SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')

# List all the files:
#   train=True, test=False will only list training files (for training)
#   train=False, test=True will only list validation files (for testing)
def list_files(labeled_train=True, labeled_test=True, unlabeled_test=False):
    r = []
    if labeled_train: 
    	r.extend(
    		[n.strip() for n in open(DATA_PATH + '/train_images.txt','r')]
    	)
    if labeled_test:
    	r.extend(
    		[n.strip() for n in open(DATA_PATH + '/test_images.txt','r')]
    	)
    if unlabeled_test:
        r.extend(
            [n.strip() for n in open(DATA_PATH + '/unlabeled_images.txt','r')]
        )
    return r

def print_for_submission(image_path, prediction):
    s = "{img},{ALB},{BET},{DOL},{LAG},{NoF},{OTHER},{SHARK},{YFT}".format(
        img=image_path.split('/')[-1],
        ALB=prediction[0],
        BET=prediction[1],
        DOL=prediction[2],
        LAG=prediction[3],
        NoF=prediction[4],
        OTHER=prediction[5],
        SHARK=prediction[6],
        YFT=prediction[7]
    )
    print s
