import librosa
import librosa.display
import matplotlib.pyplot as plt
import pylab
import sklearn
from scipy import signal
from scipy.io import wavfile
import numpy as np
from sklearn.preprocessing import minmax_scale

def create_spectrogram(filename, n_mels, n_fft):
    time_series, sample_rate = librosa.load(filename)
    spectrogram = librosa.core.stft(time_series, n_fft=n_fft, window='hamming')
    spectrogram = librosa.feature.melspectrogram(S=spectrogram,  n_mels=n_mels)
    log_spectrogram = minmax_scale(librosa.core.amplitude_to_db(np.abs(spectrogram), ref=np.max)) * 255
    return log_spectrogram
