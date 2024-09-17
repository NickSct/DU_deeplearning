import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 50
x = tf.random.uniform(shape = (N,), minval = 0, maxval = 20,seed = 42)
y = x + tf.random.normal(shape = (N,), seed = 42, )/2

inputs = tf.keras.Input(shape= (N,1), name = "input")
tf.keras.layers.BatchNormalization
hidden1 = tf.keras.layers.Dense(1, activation = "relu")(inputs)
outputs = tf.keras.layers.Dense(1, name = "output")(hidden1)
model =tf.keras.Model(inputs, outputs)
model.summary()
model.compile(loss = tf.keras.losses.mae, 
                       optimizer = tf.keras.optimizers.SGD(), 
                       metrics = ["RootMeanSquaredError"])
                    
history_0 = model.fit(tf.expand_dims(x, axis = -1),y,epochs = 100)                       

plt.scatter(x,y)
plt.show()
plt.close()

res_pd = pd.DataFrame(history_0.history)
res_pd.plot()
plt.show()
res_pd.iloc[:, 0]
