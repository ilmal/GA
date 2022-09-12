import numpy as np
import tensorflow as tf

def train(train_loader, validation_loader):

    def create_model():

      model = tf.keras.Sequential([
          tf.keras.layers.Dense(5000, activation="relu"),
          tf.keras.layers.Dense(1000, activation="relu"),
          tf.keras.layers.Dense(500, activation="relu"),
          tf.keras.layers.Dense(1, activation="sigmoid")
      ])

      model.compile(
          loss=tf.keras.losses.binary_crossentropy,
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
          metrics=[
              tf.keras.metrics.BinaryAccuracy(name="accuracy"),
              # tf.keras.metrics.Precision(name="precision"),
              # tf.keras.metrics.Recall(name="recall")
          ]
      )

      return model
    
    
    model = create_model()

    print(train_loader)

    print(validation_loader)

    model.fit(train_loader)
  
