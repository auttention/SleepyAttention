from random import shuffle
import glob
import sys
import numpy as np
from PIL import Image
import extract_spectrograms as ed
import tensorflow as tf
import re


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def createDataRecord(out_filename, addrs, labels):
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0:
            print('Data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = np.array(Image.open(addrs[i]))
        filename = re.search('\\\\(.+)\.png', addrs[i]).group(1) + '_' + str(labels[i]) + '.png'
        # Create a feature
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'filename': _bytes_feature(filename.encode('utf-8')),
            'label': _int64_feature(int(labels[i])),
            'height': _int64_feature(img.shape[0]),
            'width': _int64_feature(img.shape[1])
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def create_dataset(path_to_spectrograms, dataset_path, labels_path):
    addrs = glob.glob(path_to_spectrograms)
    csv = ed.read_labels_csv(labels_path)
    _, labels = zip(*csv)
    labels = labels[0:len(addrs)]
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
    createDataRecord(dataset_path, addrs, labels)


def create_all_datasets(spectrograms_directory="spectrograms"):
    create_dataset(spectrograms_directory + "/train/*.png", "dataset/train.tfrecords", "labels/labels_path_train.csv")
    create_dataset(spectrograms_directory + "/devel/*.png", "dataset/val.tfrecords", "labels/labels_path_devel.csv")
    create_dataset(spectrograms_directory + "/test/*.png", "dataset/test.tfrecords", "labels/labels_path_test.csv")


if __name__ == "__main__":
    create_all_datasets()