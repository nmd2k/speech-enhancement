from librosa.display import cmap
import numpy as np
from utils.tools import plot_spectrogram, plot_time_serie
import matplotlib.pyplot as plt

def visualize():
    path = 'data/train/spectrogram/noise_amp_db/'
    name = 'noise_amp_db.npy'
    data = np.load(path+name, allow_pickle=True)
    plt.imshow(data[0],extent=[0,4.2,0,8000], cmap='jet', vmin=-100, vmax=0, origin='lower', aspect='auto')
    plt.colorbar()
        # plt.imsave(f'{path}{name}_{i}.jpg', data[i], cmap='jet', origin='lower')
    plt.show()

if __name__ == '__main__':
    visualize()