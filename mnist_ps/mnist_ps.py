import tensorflow as tf

strategy = tf.distribute.ParameterServerStrategy(
    cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver()
)
global_batch_size = 256 * strategy.num_replicas_in_sync

import tensorflow_datasets as tfds
import sys

#OUTPUT_PATH = str(sys.argv[1])
INPUT_PATH = str(sys.argv[1])


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

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(global_batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(global_batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA

    ds_train = ds_train.with_options(options)
    ds_test = ds_test.with_options(options)


    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(62)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

#model.save(OUTPUT_PATH + "mnist_model")