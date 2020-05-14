import librosa
import librosa.display
import matplotlib.pyplot as plt
import pylab
import sklearn
from scipy import signal
from scipy.io import wavfile
import numpy as np
from sklearn.preprocessing import minmax_scale

"""
def create_spectrogram(filename, n_mels, n_fft):
    sample_rate, samples = wavfile.read(filename)

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nfft=n_fft, noverlap=900, nperseg=1024)

    plt.pcolormesh(times, frequencies, 10*np.log10(spectrogram))

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    return plt

"""

def create_spectrogram(filename, n_mels, n_fft):
    #pylab.axis('off')
    #pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    time_series, sample_rate = librosa.load(filename)
    spectrogram = librosa.core.stft(time_series, n_fft=n_fft, window='hamming')
    spectrogram = librosa.feature.melspectrogram(S=spectrogram,  n_mels=n_mels)
    #spectrogram = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram))
    #spectrogram = sklearn.preprocessing.scale(spectrogram, axis=1)
    #spectrogram = librosa.feature.melspectrogram(time_series, sr=sample_rate, n_mels=240, n_fft=int(0.16*sample_rate))
    #spectrogram = librosa.feature.melspectrogram(time_series, sr=sample_rate, n_mels=n_mels, n_fft=n_fft)
    log_spectrogram = minmax_scale(librosa.core.amplitude_to_db(np.abs(spectrogram), ref=np.max)) * 255
    #librosa.display.specshow(log_spectrogram, sr=sample_rate)
    return log_spectrogram
