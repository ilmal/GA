import numpy as np

def train(train_data, validation_data):

    def create_model():

      model = tf.keras.Sequential([
          tf.keras.layers.Dense(5000, activation="relu"),
          tf.keras.layers.Dense(1000, activation="relu"),
          tf.keras.layers.Dense(500, activation="relu"),
          tf.keras.layers.Dense(1, activation="sigmoid")
      ])

      model.compile(
          loss=tf.keras.losses.binary_crossentropy,
          optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
          metrics=[
              tf.keras.metrics.BinaryAccuracy(name="accuracy"),
              # tf.keras.metrics.Precision(name="precision"),
              # tf.keras.metrics.Recall(name="recall")
          ]
      )

      return model
    
    
    model.fit(train_data, validataion_data=valdataion_data)
  
