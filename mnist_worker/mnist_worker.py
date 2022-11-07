import tensorflow as tf
import pickle
import os

"""

one pod:
2727/2727 [==============================] - 93s 33ms/step - loss: 0.9097 - sparse_categorical_accuracy: 0.7421

one pod + one ps
2727/2727 [==============================] - 92s 33ms/step - loss: 0.9002 - sparse_categorical_accuracy: 0.7442

two pods
1364/1364 [==============================] - 1054s 768ms/step - loss: 1.0415 - sparse_categorical_accuracy: 0.7147

two pods + two ps
1364/1364 [==============================] - 857s 623ms/step - loss: 1.0550 - sparse_categorical_accuracy: 0.7116
1364/1364 [==============================] - 801s 587ms/step - loss: 0.6219 - sparse_categorical_accuracy: 0.8043

4 workers 4 replicas
682/682 [==============================] - 1294s 2s/step - loss: 1.2496 - sparse_categorical_accuracy: 0.6769



one pod
1396/1396 [==============================] - 1206s 862ms/step - loss: 0.7016 - sparse_categorical_accuracy: 0.7859

twp pods
349/349 [==============================] - 1630s 5s/step - loss: 1.1540 - sparse_categorical_accuracy: 0.6987

4 pods
175/175 [==============================] - 2205s 13s/step - loss: 1.4375 - sparse_categorical_accuracy: 0.6411  

"""

strategy = tf.distribute.MultiWorkerMirroredStrategy()
global_batch_size = 1000 * strategy.num_replicas_in_sync

import tensorflow_datasets as tfds
import sys

OUTPUT_PATH = str(sys.argv[2])
INPUT_PATH = str(sys.argv[1])

print("INPUT_PATH: ", INPUT_PATH)
print("OUTPUT_PATH: ", OUTPUT_PATH)

print("listdir: ", os.listdir(INPUT_PATH))

(ds_train, ds_test), ds_info = tfds.load(
    'emnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    data_dir=INPUT_PATH
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return (tf.cast(image, tf.float32) / 255., label)


print("\n", strategy.num_replicas_in_sync, "\n")

with strategy.scope():

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)

    #ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    #ds_test = ds_test.batch(global_batch_size)
    #ds_test = ds_test.cache()
    #ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA

    #dist_dataset = strategy.experimental_distribute_dataset(ds_train)


    ds_train = ds_train.with_options(options)
    #ds_test = ds_test.with_options(options)


    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(5000, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(62)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        ds_train,
        epochs=10,
        #validation_data=ds_test,
    )

with open(OUTPUT_PATH + "history.txt", "wb") as f:
    pickle.dump(history.history,f)

model.save(OUTPUT_PATH + "mnist_model")
