# Copyright astoycos@bu.edu
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import pickle


with open('pneumoia_model_history2.pkl', 'rb') as f:
	history = pickle.load(f)

epochs = list(range(1,len(history['acc']) + 1))

print(epochs)

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(epochs, history["loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Valid loss")
plt.xlabel('Epochs')
plt.legend()
plt.subplot(122)
plt.plot(epochs, history["acc"], label="Train accuracy")
plt.plot(epochs, history["val_acc"], label="Valid accuracy")
plt.xlabel('Epochs')
plt.legend()
plt.show()