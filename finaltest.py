import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD


FTRAIN = './training.csv'
FTEST = './test.csv'


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


train_X, train_y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    train_X.shape, train_X.min(), train_X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    train_y.shape, train_y.min(), train_y.max()))

X = train_X


y = train_y

model = Sequential([
        Dense(256, input_dim=X.shape[-1]),
        Activation('relu'),
        Dropout(0.4),
        Dense(y.shape[-1])
    ])
model.compile('adadelta', 'mse')


test_X, test_y = load(test=True)

model.fit(train_X, train_y,
          batch_size=32, nb_epoch=15, verbose=2)

#display matches

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

X, _ = load(test=True)
print(X)
y_pred = model.predict(X)

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)
plt.savefig('/home/aliyac1999/multi-predict.png', dpi=300)
