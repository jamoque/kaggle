from os import path
import numpy as np
import re
import config

# If you want this to work do not move this file
SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')

# List all the files:
#   train=True, test=False will only list training files (for training)
#   train=False, test=True will only list validation files (for testing)
def list_files(train=True, test=True):
    r = []
    if train: 
    	r.extend(
    		[n.strip() for n in open(DATA_PATH + '/train_images.txt','r')]
    	)
    if test:
    	r.extend(
    		[n.strip() for n in open(DATA_PATH + '/test_images.txt','r')]
    	)
    return r
