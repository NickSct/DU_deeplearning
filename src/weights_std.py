from keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

(train_x, train_y), (test_x, test_y) = mnist.load_data("data")

train_x = train_x.astype("float32") / 255.0
test_x = test_x.astype("float32") / 255.0

# Flatten the input data manually (since you commented out the Flatten layer)
train_x = train_x.reshape(-1, 28*28)  # Shape becomes (num_samples, 784)
test_x = test_x.reshape(-1, 28*28)

# One-hot encode the labels
train_y = to_categorical(train_y, 10)
test_y = to_categorical(test_y, 10)


class mnistnet(keras.models.Model):
    def __init__(self):
        super().__init__()
        #self.flatten = keras.layers.Flatten()  # Add a Flatten layer
        self.hidden1 = keras.layers.Dense(64, activation="relu")
        self.hidden2 = keras.layers.Dense(32, activation="relu")
        self.output_layer = keras.layers.Dense(10, activation="softmax")  # renamed to avoid name conflicts
        
    def call(self, inputs):
        #x = self.flatten(inputs)  # Flatten the 28x28 image
        x = self.hidden1(inputs)       # Pass through first hidden layer
        x = self.hidden2(x)       # Pass through second hidden layer
        output = self.output_layer(x)
        return output  # Output the final prediction

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Load MNIST data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Normalize input data to [0, 1] range
train_x = train_x.astype("float32") / 255.0
test_x = test_x.astype("float32") / 255.0

# Flatten the input data manually (since you commented out the Flatten layer)
train_x = train_x.reshape(-1, 28*28)  # Shape becomes (num_samples, 784)
test_x = test_x.reshape(-1, 28*28)

# One-hot encode the labels
train_y = to_categorical(train_y, 10)
test_y = to_categorical(test_y, 10)

# Create the model
model = mnistnet()

# Compile the model
model.compile(loss="categorical_crossentropy", 
              optimizer=keras.optimizers.SGD(),
              metrics=["accuracy"])

# Fit the model
model.fit(train_x, train_y, epochs=10)
