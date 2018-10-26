# Copyright astoycos@bu.edu
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
import pickle 
from keras.layers.advanced_activations import LeakyReLU

#Sources: 
#https://keras.io/optimizers/d
#https://blog.francium.tech/build-your-own-image-classifier-with-tensorflow-and-keras-dc147a15e38e
#https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
#https://wiki.python.org/moin/UsingPickle
#https://keras.io/optimizers/
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/
#https://www.analyticsvidhya.com/blog/2016/08/evolution-core-concepts-deep-learning-neural-networks/
#http://cs231n.github.io/neural-networks-3/#baby
#https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history

# dimensions of our images.
img_width, img_height = 256, 256

# directory to pull the chest x-rays from 
train_data_dir = 'data/train/'
validation_data_dir = 'data/valid/'

nb_train_samples = 5000
nb_validation_samples = 1000
epochs = 2
batch_size = 30
learning_rate = .001

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#Model architectre loosly based on ResNet5
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Conv2D(32, (3, 3)))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Conv2D(64, (5, 5)))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))


model.add(Flatten())
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Two Possible optimizers I found sgd to work better for accuracy
RMSprop = optimizers.RMSprop(lr=learning_rate)
sgd = optimizers.SGD(lr=learning_rate)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['binary_accuracy'])

#this is the augmentation configuration we will use for training
#the generator creates and passes batchs of train images 
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#this is the augmentation configuration we will use for testing:
#the generator creates and passes batchs of validation images
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
print(train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
print(validation_generator.class_indices)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#save model for use by next module 

#save model architecture 
model_json = model.to_json()
with open("pneumonia_model.json", "w") as json_file:
    json_file.write(model_json)

#save model weights 
model.save_weights('pneumonia_model_weights.h5')

#save model history 
with open('pneumoia_model_history.pkl', 'wb') as f:
	pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)





