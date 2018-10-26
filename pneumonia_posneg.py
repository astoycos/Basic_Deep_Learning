# Copyright astoycos@bu.edu
import os
import shutil
import csv
import pydicom
import png
import numpy as np
import pandas as pd
from skimage import io
from skimage import measure
from skimage.transform import resize
import itertools
from PIL import Image
from resizeimage import resizeimage
from matplotlib import pyplot as plt
import matplotlib.patches as patches

#Sources 
#https://stackoverflow.com/questions/22214949/generate-numbers-with-3-digits
#https://github.com/pydicom/pydicom/issues/352
#https://gis.stackexchange.com/questions/107568/creating-multiple-folders-named-by-a-list-in-python

#Dict to hold train Label information 
pneumonia_train = {}
pneumonia_valid = {}

number_of_train_images = 5000
number_of_valid_images = 1000

# Make directories for Testing, Verification and Training Sets 
train_data_directory_POS = 'Data/Train/pneumonia/'
train_data_directory_NEG = 'Data/Train/no_pneumonia/'
valid_data_directory_POS = 'Data/Valid/pneumonia/'
valid_data_directory_NEG = 'Data/Valid/no_pneumonia/'
test_data_directory = 'Data/Test/'

#If directory aleardy existis delete it 
if not os.path.exists(train_data_directory_POS):
	os.makedirs(train_data_directory_POS)
else: 
	shutil.rmtree(train_data_directory_POS)
	os.makedirs(train_data_directory_POS)

if not os.path.exists(train_data_directory_NEG):
	os.makedirs(train_data_directory_NEG)
else: 
	shutil.rmtree(train_data_directory_NEG)
	os.makedirs(train_data_directory_NEG)

if not os.path.exists(valid_data_directory_POS):
	os.makedirs(valid_data_directory_POS)
else: 
	shutil.rmtree(valid_data_directory_POS)
	os.makedirs(valid_data_directory_POS)

if not os.path.exists(valid_data_directory_NEG):
	os.makedirs(valid_data_directory_NEG)
else: 
	shutil.rmtree(valid_data_directory_NEG)
	os.makedirs(valid_data_directory_NEG)

if not os.path.exists(test_data_directory):
	os.makedirs(test_data_directory)
else: 
	shutil.rmtree(test_data_directory)
	os.makedirs(test_data_directory)

# Function to convert DICOM images to PNG images 
def DICOM_to_png(path,destination):
	ds = pydicom.dcmread(path)

	shape = ds.pixel_array.shape

	# Convert to float to avoid overflow or underflow losses.
	image_2d = ds.pixel_array.astype(float)

	# Rescaling grey scale between 0-255
	image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

	# Convert to uint
	image_2d_scaled = np.uint8(image_2d_scaled)

	# Write the PNG file
	with open(destination, 'wb') as png_file:
	    w = png.Writer(shape[1], shape[0], greyscale=True)
	    w.write(png_file, image_2d_scaled)

	#Resize PNG for use with model  
	with open(destination, 'r+b') as f:
	    with Image.open(f) as image:
	        cover = resizeimage.resize_contain(image, [256, 256, 1])
	        cover.save(destination, image.format)

#Create dictionaries for traning and Validation sets 
#split is dont 70% Training 30% Validation 
with open(os.path.join('stage_1_train_labels.csv'), mode = 'r') as infile: 
	reader = csv.reader(infile)
	#we want to skip the header 
	next(reader, None)

	for rows in itertools.islice(reader, 12000):
		filename = rows[0]
		disease = rows[5]

		
		if filename in pneumonia_train: 
			continue
		else:
			pneumonia_train[filename] = disease 

	for rows in itertools.islice(reader,12000,25000):
		filename = rows[0]
		disease = rows[5]

		if filename in pneumonia_valid: 
			continue
		else:
			pneumonia_valid[filename] = disease 
#counters used to show progress of preprocessing
filescounted = 0 
filescounted2 = 0 

train_number = 0
valid_number = 0 

#populate traning data directory 
for filename in itertools.islice(pneumonia_train,  number_of_train_images): 
	print("Train_files counted ", filescounted ,end="\r",flush=True)
	filescounted += 1 
	if pneumonia_train[filename] == '1':
		DICOM_to_png(("stage_1_train_images/" + filename + ".dcm"),((train_data_directory_POS) + ("pneumonia") + ("%04d" % train_number) + ".jpg"))
		train_number += 1
	else:
		DICOM_to_png(("stage_1_train_images/" + filename + ".dcm"),((train_data_directory_NEG) + ("no_pneumonia") + ("%04d" % valid_number) + ".jpg" ))
		valid_number += 1

 
train_number = 0
valid_number = 0 

print('\n')

#populate validation data directory
for filename2 in itertools.islice(pneumonia_valid, number_of_valid_images):
	print("Valid_files counted ",filescounted2,end="\r",flush=True)
	filescounted2 += 1  
	if pneumonia_valid[filename2] == '1':
		DICOM_to_png(("stage_1_train_images/" + filename2 + ".dcm"),((valid_data_directory_POS) + ("pneumonia") + ("%04d" % train_number) + ".jpg"))
		train_number += 1 
	else:
		DICOM_to_png(("stage_1_train_images/" + filename2 + ".dcm"),((valid_data_directory_NEG) + ("no_pneumonia") + ("%04d" % valid_number) + ".jpg"))
		valid_number += 1

#Populate Test data directory 
directory = os.fsencode('stage_1_test_images')

print('\n')

for idx, file in enumerate(os.listdir(directory)):
	print("Test_files counted ",idx,end="\r",flush=True)
	file = file.decode("utf-8")
	if file.endswith(".dcm"): 
		DICOM_to_png(('stage_1_test_images/' + file),(test_data_directory + ("%04d" % idx) + ".jpg"))
		continue
	else:
		continue
	
	









