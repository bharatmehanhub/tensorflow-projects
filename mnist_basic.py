import tensorflow as tf
import numpy as np
import ssl

# below is the command to create local ssl certificate on mac. Need to check if this works as is on CentOS
ssl._create_default_https_context = ssl._create_unverified_context

# Instantiating the dataset from tensorflow repo
mnist = tf.keras.datasets.mnist

# loading the dataset into memory if already present locally otherwise download the dataset and load
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizing each of the elements in input arrays as per the max size of each pixel
x_train, x_test = x_train/255.0, x_test/255.0

# instantiating and designing the model by providing input shapes,  output shapes and activations
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# generating a random output on the basis of randomly initialised weights

# log-odds https://developers.google.com/machine-learning/glossary#log-odds
model(x_train[:1]).numpy()

# random output in form of probabilities
tf.nn.softmax(model(x_train[:1]).numpy()).numpy()

# We will need to define the loss and compile the model before training it

# defining SparseCategoricalCrossentropy loss as our loss function
# The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and
# returns a scalar loss for each example.

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

"""Below commnd just implements sparsecrossentropy loss directly on test values.
We created a function above to feed it to out compiler"""
tf.keras.losses.sparse_categorical_crossentropy(y_true=y_train[:1],
                                                y_pred=model(x_train[:1]).numpy(),
                                                from_logits=True).numpy()

# compiling the model with an optimizer, loss and validation metrics
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# fitting the model on our training data
model.fit(x_train, y_train, epochs=5)

# evaluating our model on test data
model.evaluate(x_test, y_test, verbose=2)

# wrapping the trained model to attach softmax layer it it.
# This is done to productionize the model and generate outputs in form of probabilities for user inputs
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# rounding off the predicted values as required
np.round(probability_model(x_test[:1]), 2)
