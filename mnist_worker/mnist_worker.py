import tensorflow as tf
import pickle
import os

"""

(anteckningar tas bort, eller skrivs till mer senare)
docker: 264s
k8s: 1 143s


docker:
14/14 [==============================] - 156s 11s/step - loss: 3.5149 - sparse_categorical_accuracy: 0.2515

two docker:
7/7 [==============================] - 205s 25s/step - loss: 3.8088 - sparse_categorical_accuracy: 0.1549

one pod:
698/698 [==============================] - 312s 438ms/step - loss: 0.9436 - sparse_categorical_accuracy: 0.7381


DOCKER FINAL:
Epoch 1/50
698/698 [==============================] - 36s 49ms/step - loss: 2.4694 - sparse_categorical_accuracy: 0.4485
Epoch 2/50
698/698 [==============================] - 34s 48ms/step - loss: 1.4327 - sparse_categorical_accuracy: 0.6318
Epoch 3/50
698/698 [==============================] - 34s 48ms/step - loss: 1.2001 - sparse_categorical_accuracy: 0.6730
Epoch 4/50
698/698 [==============================] - 34s 48ms/step - loss: 1.1014 - sparse_categorical_accuracy: 0.6941
Epoch 5/50
698/698 [==============================] - 34s 48ms/step - loss: 1.0407 - sparse_categorical_accuracy: 0.7079
Epoch 6/50
698/698 [==============================] - 34s 48ms/step - loss: 0.9951 - sparse_categorical_accuracy: 0.7185
Epoch 7/50
698/698 [==============================] - 34s 48ms/step - loss: 0.9574 - sparse_categorical_accuracy: 0.7269
Epoch 8/50
698/698 [==============================] - 34s 48ms/step - loss: 0.9248 - sparse_categorical_accuracy: 0.7346
Epoch 9/50
698/698 [==============================] - 34s 48ms/step - loss: 0.8961 - sparse_categorical_accuracy: 0.7415
Epoch 10/50
698/698 [==============================] - 34s 48ms/step - loss: 0.8705 - sparse_categorical_accuracy: 0.7476
Epoch 11/50
698/698 [==============================] - 33s 48ms/step - loss: 0.8478 - sparse_categorical_accuracy: 0.7528
Epoch 12/50
698/698 [==============================] - 33s 48ms/step - loss: 0.8276 - sparse_categorical_accuracy: 0.7575
Epoch 13/50
698/698 [==============================] - 34s 48ms/step - loss: 0.8096 - sparse_categorical_accuracy: 0.7616
Epoch 14/50
698/698 [==============================] - 34s 48ms/step - loss: 0.7935 - sparse_categorical_accuracy: 0.7651
Epoch 15/50
698/698 [==============================] - 34s 48ms/step - loss: 0.7789 - sparse_categorical_accuracy: 0.7685
Epoch 16/50
698/698 [==============================] - 34s 48ms/step - loss: 0.7657 - sparse_categorical_accuracy: 0.7715
Epoch 17/50
698/698 [==============================] - 33s 48ms/step - loss: 0.7536 - sparse_categorical_accuracy: 0.7741
Epoch 18/50
698/698 [==============================] - 34s 49ms/step - loss: 0.7426 - sparse_categorical_accuracy: 0.7765
Epoch 19/50
698/698 [==============================] - 34s 48ms/step - loss: 0.7325 - sparse_categorical_accuracy: 0.7789
Epoch 20/50
698/698 [==============================] - 34s 49ms/step - loss: 0.7232 - sparse_categorical_accuracy: 0.7811
Epoch 21/50
698/698 [==============================] - 34s 49ms/step - loss: 0.7146 - sparse_categorical_accuracy: 0.7831
Epoch 22/50
698/698 [==============================] - 34s 48ms/step - loss: 0.7067 - sparse_categorical_accuracy: 0.7849
Epoch 23/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6993 - sparse_categorical_accuracy: 0.7866
Epoch 24/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6924 - sparse_categorical_accuracy: 0.7881
Epoch 25/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6860 - sparse_categorical_accuracy: 0.7896
Epoch 26/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6799 - sparse_categorical_accuracy: 0.7909
Epoch 27/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6743 - sparse_categorical_accuracy: 0.7922
Epoch 28/50
698/698 [==============================] - 33s 48ms/step - loss: 0.6689 - sparse_categorical_accuracy: 0.7934
Epoch 29/50
698/698 [==============================] - 34s 49ms/step - loss: 0.6639 - sparse_categorical_accuracy: 0.7946
Epoch 30/50
698/698 [==============================] - 34s 49ms/step - loss: 0.6592 - sparse_categorical_accuracy: 0.7957
Epoch 31/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6547 - sparse_categorical_accuracy: 0.7968
Epoch 32/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6504 - sparse_categorical_accuracy: 0.7977
Epoch 33/50
698/698 [==============================] - 34s 49ms/step - loss: 0.6464 - sparse_categorical_accuracy: 0.7987
Epoch 34/50
698/698 [==============================] - 33s 48ms/step - loss: 0.6425 - sparse_categorical_accuracy: 0.7995
Epoch 35/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6389 - sparse_categorical_accuracy: 0.8003
Epoch 36/50
698/698 [==============================] - 34s 49ms/step - loss: 0.6354 - sparse_categorical_accuracy: 0.8011
Epoch 37/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6321 - sparse_categorical_accuracy: 0.8018
Epoch 38/50
698/698 [==============================] - 33s 48ms/step - loss: 0.6289 - sparse_categorical_accuracy: 0.8025
Epoch 39/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6258 - sparse_categorical_accuracy: 0.8033
Epoch 40/50
698/698 [==============================] - 33s 48ms/step - loss: 0.6229 - sparse_categorical_accuracy: 0.8040
Epoch 41/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6202 - sparse_categorical_accuracy: 0.8046
Epoch 42/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6175 - sparse_categorical_accuracy: 0.8052
Epoch 43/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6150 - sparse_categorical_accuracy: 0.8058
Epoch 44/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6125 - sparse_categorical_accuracy: 0.8063
Epoch 45/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6102 - sparse_categorical_accuracy: 0.8069
Epoch 46/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6079 - sparse_categorical_accuracy: 0.8074
Epoch 47/50
698/698 [==============================] - 34s 48ms/step - loss: 0.6057 - sparse_categorical_accuracy: 0.8078
Epoch 48/50
698/698 [==============================] - 33s 47ms/step - loss: 0.6036 - sparse_categorical_accuracy: 0.8083
Epoch 49/50
698/698 [==============================] - 33s 48ms/step - loss: 0.6016 - sparse_categorical_accuracy: 0.8087
Epoch 50/50
698/698 [==============================] - 33s 48ms/step - loss: 0.5997 - sparse_categorical_accuracy: 0.8092

K8S FINAL:
Epoch 1/50
175/175 [==============================] - 87s 435ms/step - loss: 3.5251 - sparse_categorical_accuracy: 0.1769
Epoch 2/50
175/175 [==============================] - 37s 211ms/step - loss: 2.5300 - sparse_categorical_accuracy: 0.4580
Epoch 3/50
175/175 [==============================] - 37s 212ms/step - loss: 1.9972 - sparse_categorical_accuracy: 0.5417
Epoch 4/50
175/175 [==============================] - 38s 215ms/step - loss: 1.6797 - sparse_categorical_accuracy: 0.5912
Epoch 5/50
175/175 [==============================] - 37s 209ms/step - loss: 1.4851 - sparse_categorical_accuracy: 0.6224
Epoch 6/50
175/175 [==============================] - 48s 272ms/step - loss: 1.3583 - sparse_categorical_accuracy: 0.6446
Epoch 7/50
175/175 [==============================] - 41s 233ms/step - loss: 1.2735 - sparse_categorical_accuracy: 0.6600
Epoch 8/50
175/175 [==============================] - 36s 207ms/step - loss: 1.2135 - sparse_categorical_accuracy: 0.6713
Epoch 9/50
175/175 [==============================] - 36s 206ms/step - loss: 1.1657 - sparse_categorical_accuracy: 0.6806
Epoch 10/50
175/175 [==============================] - 37s 209ms/step - loss: 1.1293 - sparse_categorical_accuracy: 0.6890
Epoch 11/50
175/175 [==============================] - 36s 207ms/step - loss: 1.0991 - sparse_categorical_accuracy: 0.6955
Epoch 12/50
175/175 [==============================] - 36s 204ms/step - loss: 1.0763 - sparse_categorical_accuracy: 0.7008
Epoch 13/50
175/175 [==============================] - 36s 206ms/step - loss: 1.0530 - sparse_categorical_accuracy: 0.7064
Epoch 14/50
175/175 [==============================] - 36s 207ms/step - loss: 1.0351 - sparse_categorical_accuracy: 0.7106
Epoch 15/50
175/175 [==============================] - 45s 257ms/step - loss: 1.0169 - sparse_categorical_accuracy: 0.7145
Epoch 16/50
175/175 [==============================] - 40s 226ms/step - loss: 1.0018 - sparse_categorical_accuracy: 0.7178
Epoch 17/50
175/175 [==============================] - 41s 231ms/step - loss: 0.9877 - sparse_categorical_accuracy: 0.7215
Epoch 18/50
175/175 [==============================] - 37s 208ms/step - loss: 0.9732 - sparse_categorical_accuracy: 0.7249
Epoch 19/50
175/175 [==============================] - 37s 210ms/step - loss: 0.9592 - sparse_categorical_accuracy: 0.7283
Epoch 20/50
175/175 [==============================] - 37s 211ms/step - loss: 0.9470 - sparse_categorical_accuracy: 0.7314
Epoch 21/50
175/175 [==============================] - 36s 205ms/step - loss: 0.9352 - sparse_categorical_accuracy: 0.7339
Epoch 22/50
175/175 [==============================] - 36s 205ms/step - loss: 0.9227 - sparse_categorical_accuracy: 0.7370
Epoch 23/50
175/175 [==============================] - 36s 207ms/step - loss: 0.9121 - sparse_categorical_accuracy: 0.7394
Epoch 24/50
175/175 [==============================] - 36s 204ms/step - loss: 0.9010 - sparse_categorical_accuracy: 0.7420
Epoch 25/50
175/175 [==============================] - 42s 239ms/step - loss: 0.8917 - sparse_categorical_accuracy: 0.7442
Epoch 26/50
175/175 [==============================] - 47s 266ms/step - loss: 0.8813 - sparse_categorical_accuracy: 0.7465
Epoch 27/50
175/175 [==============================] - 37s 211ms/step - loss: 0.8716 - sparse_categorical_accuracy: 0.7492
Epoch 28/50
175/175 [==============================] - 41s 231ms/step - loss: 0.8630 - sparse_categorical_accuracy: 0.7511
Epoch 29/50
175/175 [==============================] - 36s 202ms/step - loss: 0.8562 - sparse_categorical_accuracy: 0.7529
Epoch 30/50
175/175 [==============================] - 37s 208ms/step - loss: 0.8474 - sparse_categorical_accuracy: 0.7547
Epoch 31/50
175/175 [==============================] - 36s 206ms/step - loss: 0.8385 - sparse_categorical_accuracy: 0.7570
Epoch 32/50
175/175 [==============================] - 36s 203ms/step - loss: 0.8306 - sparse_categorical_accuracy: 0.7586
Epoch 33/50
175/175 [==============================] - 36s 207ms/step - loss: 0.8218 - sparse_categorical_accuracy: 0.7607
Epoch 34/50
175/175 [==============================] - 36s 203ms/step - loss: 0.8151 - sparse_categorical_accuracy: 0.7622
Epoch 35/50
175/175 [==============================] - 36s 204ms/step - loss: 0.8073 - sparse_categorical_accuracy: 0.7639
Epoch 36/50
175/175 [==============================] - 36s 205ms/step - loss: 0.8023 - sparse_categorical_accuracy: 0.7650
Epoch 37/50
175/175 [==============================] - 52s 298ms/step - loss: 0.7965 - sparse_categorical_accuracy: 0.7664
Epoch 38/50
175/175 [==============================] - 41s 233ms/step - loss: 0.7908 - sparse_categorical_accuracy: 0.7679
Epoch 39/50
175/175 [==============================] - 35s 202ms/step - loss: 0.7852 - sparse_categorical_accuracy: 0.7689
Epoch 40/50
175/175 [==============================] - 36s 206ms/step - loss: 0.7795 - sparse_categorical_accuracy: 0.7702
Epoch 41/50
175/175 [==============================] - 37s 210ms/step - loss: 0.7735 - sparse_categorical_accuracy: 0.7715
Epoch 42/50
175/175 [==============================] - 37s 208ms/step - loss: 0.7688 - sparse_categorical_accuracy: 0.7727
Epoch 43/50
175/175 [==============================] - 36s 207ms/step - loss: 0.7644 - sparse_categorical_accuracy: 0.7740
Epoch 44/50
175/175 [==============================] - 36s 204ms/step - loss: 0.7586 - sparse_categorical_accuracy: 0.7750
Epoch 45/50
175/175 [==============================] - 36s 207ms/step - loss: 0.7544 - sparse_categorical_accuracy: 0.7762
Epoch 46/50
175/175 [==============================] - 44s 253ms/step - loss: 0.7505 - sparse_categorical_accuracy: 0.7769
Epoch 47/50
175/175 [==============================] - 43s 246ms/step - loss: 0.7460 - sparse_categorical_accuracy: 0.7776
Epoch 48/50
175/175 [==============================] - 39s 223ms/step - loss: 0.7430 - sparse_categorical_accuracy: 0.7786
Epoch 49/50
175/175 [==============================] - 35s 201ms/step - loss: 0.7403 - sparse_categorical_accuracy: 0.7790
Epoch 50/50
175/175 [==============================] - 36s 205ms/step - loss: 0.7333 - sparse_categorical_accuracy: 0.7807

"""

strategy = tf.distribute.MultiWorkerMirroredStrategy()
global_batch_size = 1000 * strategy.num_replicas_in_sync

import tensorflow_datasets as tfds
import sys
import json

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

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = \
    tf.data.experimental.AutoShardPolicy.DATA

ds_train = ds_train.with_options(options)


with strategy.scope():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(62)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=(OUTPUT_PATH + "logs"), 
    histogram_freq=1, 
    write_steps_per_second=True
)


history = model.fit(
    ds_train,
    epochs=50,
    callbacks=[tensorboard_callback]
    #validation_data=ds_test,
)

json.dump(history.history, open(OUTPUT_PATH + "history.txt", 'w'))

# with open(OUTPUT_PATH + "history.txt", "wb") as f:
#     print("saving pickle")
#     pickle.dump(history,f)
#     print("pickle saved")

model.save(OUTPUT_PATH + "mnist_model")
