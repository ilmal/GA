print("1")
print("1")
print("1")
print("1")
print("1")
print("1")
print("1")
print("1")
print("1")
print("1")
print("1")
print("1")
print("1")
import tensorflow as tf
import sys
import os
import json

# {'cluster': {'ps': ['tfjob-mnistvh4ds-ps-0.kubeflow.svc:2222'], 'worker': ['tfjob-mnistvh4ds-worker-0.kubeflow.svc:2222']}, 'task': {'type': 'ps', 'index': 0}, 'environment': 'cloud'}

tf_config = json.loads(os.environ.get("TF_CONFIG"))

# config
global_batch_size = 5000  #  As big as will fit on my gpu

OUTPUT_PATH = str(sys.argv[2])
INPUT_PATH = str(sys.argv[1])

print("2")

def server():
    print("THIS IS A SERVER")
    server = tf.distribute.Server(
        tf_config["cluster"],
        job_name=tf_config["task"]["type"],
        task_index=tf_config["task"]["index"],
        protocol="grpc")
    server.join()

def controller():
    print("THIS IS NOT A SERVER")

    # load mnist data set
    import tensorflow_datasets as tfds
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

    # def dataset_fn():
    #     return ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)

    # input = tf.keras.utils.experimental.DatasetCreator(dataset_fn=dataset_fn)

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE).shuffle(20).repeat()
    ds_train = ds_train.batch(global_batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA

    ds_train = ds_train.with_options(options)

    # model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(62)
    # ])

    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(0.0001),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    # )


    variable_partitioner = (
        tf.distribute.experimental.partitioners.MinSizePartitioner(
            min_shard_bytes=(256 << 10),
            max_shards=2))

    strategy = tf.distribute.ParameterServerStrategy(
        cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver(),
        variable_partitioner=variable_partitioner
    )

    with strategy.scope():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(64, activation='relu'),
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
        epochs=10,
        steps_per_epoch=100
        #validation_data=ds_test,
    )



if tf_config["task"]["type"] == "ps":
    print("3")
    server()
elif tf_config["task"]["type"] == "worker" and tf_config["task"]["index"] != 0:
    print("4")
    controller()

if tf_config["task"]["type"] == "worker" and tf_config["task"]["index"] == 0:
    controller()