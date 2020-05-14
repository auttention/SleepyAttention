import PIL
import csv
import glob
import re
from PIL import Image
import audio_converter as ac
import cv2
import os
import create_dataset
import numpy as np


#PATH_TO_LABELS = "labels\labels.csv"
# PATH_TO_LABELS = r"C:\Users\PawelWinokurow\ComParE2019_ContinuousSleepiness\ComParE2019_ContinuousSleepiness\lab\labels.csv"
# PATH_TO_LABELS = "/Users/pawelwinokurow/Desktop/ComParE2019_ContinuousSleepiness/lab/labels.csv"
PATH_TO_WAV = "D:\ComParE2019_ContinuousSleepiness\wav"
# PATH_TO_WAV = r"C:\Users\PawelWinokurow\ComParE2019_ContinuousSleepiness\ComParE2019_ContinuousSleepiness\wav"
# PATH_TO_WAV = "/Users/pawelwinokurow/Desktop/ComParE2019_ContinuousSleepiness/wav"
PATH_TO_LABELS_TRAIN = "labels/labels_path_train.csv"
PATH_TO_LABELS_DEVEL = "labels/labels_path_devel.csv"
PATH_TO_LABELS_TEST = "labels/labels_path_test.csv"



def read_names_and_labels(path_to_labels):
    list_train = []
    list_devel = []
    list_test = []
    with open(path_to_labels) as labels_csv:
        for line in labels_csv:
            groups = re.search(r"(train_\d+)\.wav,([1-9])", line)
            if groups:
                list_train += [[groups.group(1), groups.group(2)]]
            groups = re.search(r"(devel_\d+)\.wav,([1-9])", line)
            if groups:
                list_devel += [[groups.group(1), groups.group(2)]]
            groups = re.search(r"(test_\d+)\.wav,([1-9])", line)
            if groups:
                list_test += [[groups.group(1), groups.group(2)]]
    return list_train, list_devel, list_test


def read_labels_csv(labels_path):
    list = []
    with open(labels_path) as labels_csv:
        reader = csv.reader(labels_csv)
        for line in reader:
            #            if line[1] != '?':
            #                line[1] = int(line[1])
            list += [line]
    return list


def write_csv(list, filename, path_to_spectrograms):
    with open(filename, mode='w', newline='') as labels_csv:
        writer = csv.writer(labels_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in list:
            writer.writerow([os.path.join(path_to_spectrograms, line[0]) + ".png", line[1]])


def create_spectrograms_and_save(n_fft, n_mels, from_directory, to_directory, list):
    os.makedirs(to_directory, exist_ok=True)
    for line in list:
        filename = os.path.join(to_directory, line[0]) + ".png"
        if not os.path.isfile(filename):
            spectrogram = ac.create_spectrogram(os.path.join(from_directory, line[0]) + ".wav", n_fft=n_fft, n_mels=n_mels)
            spectrogram = Image.fromarray(spectrogram).convert("L")
            spectrogram.save(filename)
            #spectrogram.savefig(filename, bbox_inches=None, pad_inches=0)
            #spectrogram.close()
            #img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype('float32')
            #resized_img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
            #cv2.imwrite(filename, resized_img)


def extract_spectrograms(n_fft, n_mels, from_directory, to_directory, path_to_labels):
    list_train, list_devel, list_test = read_names_and_labels(path_to_labels)
    write_csv(list_train, PATH_TO_LABELS_TRAIN, os.path.join(to_directory, "train"))
    write_csv(list_devel, PATH_TO_LABELS_DEVEL, os.path.join(to_directory, "devel"))
    write_csv(list_test, PATH_TO_LABELS_TEST, os.path.join(to_directory, "test"))
    create_spectrograms_and_save(n_fft, n_mels, from_directory, os.path.join(to_directory, "train"), list_train)
    create_spectrograms_and_save(n_fft, n_mels, from_directory, os.path.join(to_directory, "devel"), list_devel)
    create_spectrograms_and_save(n_fft, n_mels, from_directory, os.path.join(to_directory, "test"), list_test)
    normalize_spectrograms(to_directory)


def normalize_spectrograms(directory):
    addrs = glob.glob(directory + "/train/*.png") + glob.glob(directory + "/devel/*.png") + glob.glob(directory + "/test/*.png")
    min = 100000000
    for addr in addrs:
        img = np.asarray(PIL.Image.open(addr))
        if img.shape[1] < min:
            min = img.shape[1]
    print(min)
    for addr in addrs:
        img = np.asarray(PIL.Image.open(addr))
        #if img.shape[1] < 160:
        #   pad_size = 160 - img.shape[1]
        #    img = np.pad(img, ((0, 0), (0, pad_size)), "constant", constant_values=(0))
        #else:
        img = img[:,:min]
        img = Image.fromarray(img)
        img.save(addr)
        img = cv2.imread(addr, cv2.IMREAD_GRAYSCALE).astype('float32')
        resized_img = cv2.resize(img, dsize=(min, 120), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(addr, resized_img)

