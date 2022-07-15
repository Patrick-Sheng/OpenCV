import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import LSTM
from keras import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# print("Tensorflow version:", tf.__version__)

actions = np.array(['cSitting', 'cStanding'])
label_map = {label: num for num, label in enumerate(actions)}
print(label_map)

video, labels = [], []
data_number = 10
data_length = 10
file_path = os.path.join('.dataCollection')

for action in actions:
    for sequence in range(data_number):
        temp_array = []
        for no_frame in range(data_length):
            # try:
            picture = np.load(file_path + "/" + action + "/" + str(sequence) + "/" + str(no_frame) + ".npy")
            temp_array.append(picture)
            # except:
            #     print('fail')
            #     pass
        video.append(temp_array)
        labels.append(label_map[action])

np_video = np.array(video)
np_labels = np.array(labels)

# print(np_video.shape)
# print(np_labels.shape)

labels_array = to_categorical(labels).astype(int)
# print(labels_array.shape)

# Splitting data using train_test_split function, assign 20% data to test.
x_train, x_test, y_train, y_test = train_test_split(np_video, labels_array, test_size= 0.2)
# print(x_test.shape)
# print(x_train.shape)

# Creating Sequential model, adding LSTM layers and Dense layers
model = tf.keras.Sequential()
model.add()

# model.summary()

# model.compile()
# model.fit()
# loss, acc = model.evaluate()
# ??? = model.predict()


# X = np.array(video)
# Y =
# print(X)
