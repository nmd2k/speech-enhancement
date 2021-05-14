import os
import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

# DATA_PATH = './data/train/spectrogram/'

# filenames = os.listdir(DATA_PATH)
# print(filenames)


# data = np.load(DATA_PATH+'noisy_voice_pha_db.npy', allow_pickle=True)
# plt.imshow(data[0],extent=[0,4.2,0,48000], cmap='jet', vmin=-100, vmax=0, origin='lower', aspect='auto')
# plt.colorbar()
# plt.title('noise_amp_db.npy')

# # fig, axs = plt.subplots(2, 3)
# # index = 0
# # for i in range(2):
# #     for j in range(3):
# #         data = np.load(DATA_PATH+filenames[index], allow_pickle=True)
# #         axs[i, j].imshow(data[0],extent=[0,4.2,0,48000], cmap='jet', vmin=-100, vmax=0, origin='lower', aspect='auto')
# #         # axs[i, j].colorbar()
# #         axs[i, j].set_title(filenames[index])
# #         index += 1

# plt.show()

def change_name(path):
    """Change the duplicate file name"""
    os.chdir(path)
    filenames = os.listdir()

    print(filenames)
    for dir in filenames:
        for file in os.listdir(dir):
            os.rename(dir+'/'+file, f'{dir}_{file}')

def plot_spectrogram(stftaudio_magnitude_db, sample_rate, hop_length_fft) :
    """This function plots a spectrogram"""
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(stftaudio_magnitude_db, x_axis='time', y_axis='linear',
                             sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    title = 'hop_length={},  time_steps={},  fft_bins={}  (2D resulting shape: {})'
    plt.title(title.format(hop_length_fft,
                           stftaudio_magnitude_db.shape[1],
                           stftaudio_magnitude_db.shape[0],
                           stftaudio_magnitude_db.shape));
    return

def plot_time_serie(audio,sample_rate):
    """This function plots the audio as a time serie"""
    plt.figure(figsize=(12, 6))
    #plt.ylim(-0.05, 0.05)
    plt.title('Audio')
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    librosa.display.waveplot(audio, sr=sample_rate)
    return