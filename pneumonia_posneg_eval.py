# Copyright astoycos@bu.edu
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
from keras.models import model_from_json
from keras.preprocessing import image
import pickle
import random

from matplotlib import pyplot as plt
import matplotlib.patches as patches

import pickle

#Sources
#https://matplotlib.org/index.html
#https://wiki.python.org/moin/UsingPickle
#https://towardsdatascience.com/basics-of-image-classification-with-keras-43779a299c8b
#https://www.pythoncentral.io/how-to-generate-a-random-number-in-python/
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history

#open model history 
with open('pneumoia_model_history10.pkl', 'rb') as f:
	history = pickle.load(f)
print(history)

epochs = list(range(1,len(history['binary_accuracy']) + 1))

print(epochs)

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(epochs, history["loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Valid loss")
plt.xlabel('Epochs')
plt.legend()
plt.subplot(122)
plt.plot(epochs, history["binary_accuracy"], label="Train accuracy")
plt.plot(epochs, history["val_binary_accuracy"], label="Valid accuracy")
plt.xlabel('Epochs')
plt.legend()
plt.show()

# load json and create model
json_file = open('pneumonia_model10.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('pneumonia_model_weights10.h5')
print("Loaded model from disk")
 
# evaluate loaded model on test data
sgd = optimizers.SGD(lr=.0001)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])

plt.figure(figsize=(54,10))

#plot model predictions on test set 
for x in range(9):
	rand = random.randint(0,1001)
	filename = ("%04d" % rand) + ".jpg"
	img = image.load_img("data/Test/" + filename, target_size=(256,256,3))
	img2 = image.img_to_array(img)
	test_img = img2.reshape((1,256,256,3))
	
	print(model.predict(test_img))
	img_class = model.predict_classes(test_img)
	img_prob = model.predict_proba(test_img)
	
	prediction = img_class[0]
	classname = img_class[0]
	
	print("Class: ",classname)

	
	plt.subplot(int("52" + str(x + 1)))
	plt.imshow(img)
	plt.title(classname)

plt.tight_layout()
plt.show()
