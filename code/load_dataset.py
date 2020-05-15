import tensorflow as tf

directory = "dataset/"

def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        'filename': tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
        "height": tf.FixedLenFeature([], tf.int64),
        "width": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    filename = parsed['filename']
    height = tf.cast(parsed['height'], tf.int32)
    width = tf.cast(parsed['width'], tf.int32)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.reshape(image, tf.stack([width, height]))
    #image = tf.reshape(image, shape=[431, 96])
    image = tf.cast(image, tf.float32)
    return {'image': image}, {'filename': filename}, parsed["label"]


def input_fn(filenames, batch_size=64):
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.repeat(1)
    dataset = dataset.map(parser)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=2)
    return dataset


def train_input_fn(batch_size=64):
    return input_fn(filenames=[directory + "train.tfrecords"], batch_size=batch_size)


def test_input_fn(batch_size=64):
    return input_fn(filenames=[directory + "test.tfrecords"], batch_size=batch_size)


def val_input_fn(batch_size=64):
    return input_fn(filenames=[directory + "val.tfrecords"], batch_size=batch_size)


