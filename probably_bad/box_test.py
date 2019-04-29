import tensorflow as tf
from tensorflow.keras import layers
import random
from sklearn.metrics import mean_squared_error

# model = tf.keras.Sequential()
# # Adds a densely-connected layer with 64 units to the model:
# model.add(layers.Dense(64, activation='relu'))
# # Add another:
# model.add(layers.Dense(64, activation='relu'))
# # Add a softmax layer with 10 output units:
# model.add(layers.Dense(1, activation='softmax'))

# # Create a sigmoid layer:
# layers.Dense(64, activation='sigmoid')
# # Or:
# layers.Dense(64, activation=tf.sigmoid)

# # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
# layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
# layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# # A linear layer with a kernel initialized to a random orthogonal matrix:
# layers.Dense(64, kernel_initializer='orthogonal')

# # A linear layer with a bias vector initialized to 2.0s:
# layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(64,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 4 output units:
layers.Dense(4, activation='softmax')])

# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
              loss='mse',       # mean squared error
              metrics=['mape'])  # mean absolute error

# # Configure a model for categorical classification.
# model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
#               loss=tf.keras.losses.categorical_crossentropy,
#               metrics=[tf.keras.metrics.categorical_accuracy])


import numpy as np

train_info = []
train_img = []
for i in range(100000):
    a = np.zeros([8,8])
    y = random.randint(0,7)
    x = random.randint(0,7)
    h = random.randint(0,7-y)
    w = random.randint(0,7-x)
    for t in range(w):
        for k in range(h):
            a[y+k][x+t] = 1
    # train_info.append(x)
    train_info.append([x+1,y+1,w+1,h+1])
    train_img.append(a.reshape(64))
train_img = np.array(train_img)
train_info = np.array(train_info)


test_info = []
test_img = []
for i in range(100):
    a = np.zeros([8,8])
    y = random.randint(0,7)
    x = random.randint(0,7)
    h = random.randint(0,7-y)
    w = random.randint(0,7-x)
    for t in range(w):
        for k in range(h):
            a[y+k][x+t] = 1
    # test_info.append(x)
    test_info.append([x+1,y+1,w+1,h+1])
    test_img.append(a.reshape(64))
test_img = np.array(test_img)
test_info = np.array(test_info)


model.fit(train_img, train_info, epochs=1000, batch_size=100)

result = model.predict(test_img, batch_size=32)

print(test_info)
print(result)
print(test_info - result)
print(mean_squared_error(test_info, result))

