import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras import optimizers

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras

import pickle

# code for loading the data is from Florian LE BOURDAIS (http://flothesof.github.io/convnet-face-keypoint-detection.html)

df = pd.read_csv('./training.csv')

def string2image(string):
    """Converts a string to a numpy array."""
    return np.array([int(item) for item in string.split()]).reshape((96, 96))

fully_annotated = df.dropna()
# fully_annotated = df.fillna(df.mean()) #https://stackoverflow.com/questions/18689823/pandas-dataframe-replace-nan-values-with-average-of-columns
# print("Begin filling na values")
# fully_annotated = df.apply(lambda x: x.fillna(x.mean()),axis=0)
# print("Done filling na values")
X = np.stack([string2image(string) for string in fully_annotated['Image']]).astype(np.float)[:, :, :, np.newaxis]

y = np.vstack(fully_annotated[fully_annotated.columns[:-1]].values)

X_train = X / 255.

output_pipe = make_pipeline(
    MinMaxScaler(feature_range=(-1, 1))
)

y_train = output_pipe.fit_transform(y)

#save the pipeline for later to transform the model output back to real coords
with open('my_sklearn_objects.pkl', 'wb') as f:
    pickle.dump((output_pipe), f)

# end of Florian LE BOURDAIS load code

# Florian LE BOURDAIS code for plotting points - modified for testing with google cloud engine
def plot_faces_with_keypoints_and_predictions(model, nrows=5, ncols=5, model_input='flat'):
    """Plots sampled faces with their truth and predictions."""
    selection = np.random.choice(np.arange(X.shape[0]), size=(nrows*ncols), replace=False)
    fig, axes = plt.subplots(figsize=(10, 10), nrows=nrows, ncols=ncols)
    for ind, ax in zip(selection, axes.ravel()):
        img = X_train[ind, :, :, 0]
        if model_input == 'flat':
            predictions = model.predict(img.reshape(1, -1))
        else:
            predictions = model.predict(img[np.newaxis, :, :, np.newaxis])
        xy_predictions = output_pipe.inverse_transform(predictions).reshape(15, 2)
        ax.imshow(img, cmap='gray')
        ax.plot(xy_predictions[:, 0], xy_predictions[:, 1], 'bo')
        ax.axis('off')
    plt.savefig('/home/aliyac1999/better-predict.png', dpi=300)

#CNN model structure from Peter Skvarenina for Medium.com (https://towardsdatascience.com/detecting-facial-features-using-deep-learning-2e23c8660a7a)

model = Sequential()
# input layer
model.add(BatchNormalization(input_shape=(96, 96, 1)))
model.add(Conv2D(24, (5, 5), kernel_initializer='he_normal')) #draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) #prevent overfitting
model.add(Dropout(0.2))
# layer 2
model.add(Conv2D(36, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
# layer 3
model.add(Conv2D(48, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
# layer 4
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
# layer 5
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Flatten())
# layer 6
model.add(Dense(500, activation="relu"))
# layer 7
model.add(Dense(90, activation="relu"))
# layer 8
model.add(Dense(30))

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
epochs = 50

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=90,
    horizontal_flip=True)
datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=20),
                steps_per_epoch=len(X_train) / 20, shuffle=True, 
                epochs=epochs)

model.save('/home/aliyac1999/finalmodel.h5')

#this line loads the model
#model = keras.models.load_model("finalmodel.h5")

#this saves 16 random faces with keypoints
plot_faces_with_keypoints_and_predictions(model, model_input='2d')