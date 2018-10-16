import os
import csv
import random
import pydicom
import numpy as np
import pandas as pd
from skimage import io
from skimage import measure
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
import matplotlib.patches as patches

#Dict to hold train Label information 
pheumonia_bboxes = {}

with open(os.path.join('stage_1_train_labels.csv'), mode = 'r') as infile: 
	reader = csv.reader(infile)
	#we want to skip the header 
	next(reader, None)

	for rows in reader: 
		filename = rows[0]
		bbox_loc = rows[1:5]
		disease = rows[5]

	if(diesase == '1'):
		for i in location:
			location = [Int(i)]

		if filename in pneumonia_bboxes: 
			pneumonia_bboxes[filename].append(location)
		else:
			pneumonia_bboxes[filename].append
