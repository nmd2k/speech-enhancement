# Author: github.com/vbelz/Speech-enhancement
from model.model import UNet, UNet_ResNet
import torch
from torchvision import transforms
from model.config import *
import os
import librosa
import librosa.display
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def convertaudio_to_wav(filename):
    """Converter from other audio format to wav
    """
    AudioSegment.converter = "E:\Systems\miniconda3\envs\ml\Lib\site-packages\ffmpeg"
    if filename[-3:] == 'mp3':
        sound = AudioSegment.from_mp3(filename)
    if filename[-3:] == 'ogg':
        sound = AudioSegment.from_ogg(filename)
    if filename[-3:] == 'mp3':
        sound = AudioSegment.from_flv(filename)
    else: return

    sound.export(UPLOAD_FOLDER+filename[:-3]+'wav', format="wav")
    return 

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):
    """This function take an audio and split into several frame
       in a numpy matrix of size (nb_frame,frame_length)"""

    sequence_sample_length = sound_data.shape[0]

    sound_data_list = [sound_data[start:start + frame_length] for start in range(
    0, sequence_sample_length - frame_length + 1, hop_length_frame)]  # get sliding windows
    sound_data_array = np.vstack(sound_data_list)

    return sound_data_array


def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
    """This function take audio files of a directory and merge them
    in a numpy matrix of size (nb_frame,frame_length) for a sliding window of size hop_length_frame"""

    list_sound_array = []

    for file in list_audio_files:
        # open the audio file
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
        total_duration = librosa.get_duration(y=y, sr=sr)

        if (total_duration >= min_duration):
            list_sound_array.append(audio_to_audio_frame_stack(
                y, frame_length, hop_length_frame))
        else:
            print(
                f"The following file {os.path.join(audio_dir,file)} is below the min duration")

    return np.vstack(list_sound_array)

def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio):
    """This function takes an audio and convert into spectrogram,
       it returns the magnitude in dB and the phase"""

    stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)

    stftaudio_magnitude_db = librosa.amplitude_to_db(
        stftaudio_magnitude, ref=np.max)

    return stftaudio_magnitude_db, stftaudio_phase


def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
    """This function takes as input a numpy audi of size (nb_frame,frame_length), and return
    a numpy containing the matrix spectrogram for amplitude in dB and phase. It will have the size
    (nb_frame,dim_square_spec,dim_square_spec)"""

    nb_audio = numpy_audio.shape[0]

    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

    for i in range(nb_audio):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            n_fft, hop_length_fft, numpy_audio[i])

    return m_mag_db, m_phase


def magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):
    """This functions reverts a spectrogram to an audio"""

    stftaudio_magnitude_rev = librosa.db_to_amplitude(stftaudio_magnitude_db, ref=1.0)

    # taking magnitude and phase of audio
    audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
    audio_reconstruct = librosa.core.istft(audio_reverse_stft, hop_length=hop_length_fft, length=frame_length)

    return audio_reconstruct

def matrix_spectrogram_to_numpy_audio(m_mag_db, m_phase, frame_length, hop_length_fft)  :
    """This functions reverts the matrix spectrograms to numpy audio"""

    list_audio = []

    nb_spec = m_mag_db.shape[0]

    for i in range(nb_spec):

        audio_reconstruct = magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, m_mag_db[i], m_phase[i])
        list_audio.append(audio_reconstruct)

    return np.vstack(list_audio)


def scaled_in(matrix_spec):
    "global scaling apply to noisy voice spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec

def scaled_ou(matrix_spec):
    "global scaling apply to noise models spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec - 6)/82
    return matrix_spec

def inv_scaled_in(matrix_spec):
    "inverse global scaling apply to noisy voices spectrograms"
    matrix_spec = matrix_spec * 50 - 46
    return matrix_spec

def inv_scaled_ou(matrix_spec):
    "inverse global scaling apply to noise models spectrograms"
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec

def model_denoising(filename, model_type='Unet'):
    audio = audio_files_to_numpy(UPLOAD_FOLDER, [filename], SAMPLE_RATE,
                                FRAME_LENGTH, HOP_LENGTH_FRAME, MIN_DURATION)

    # Squared spectrogram dimensions
    dim_square_spec = int(N_FFT / 2) + 1

    m_amp_db,  m_pha = numpy_audio_to_matrix_spectrogram(
                audio, dim_square_spec, N_FFT, HOP_LENGTH_FFT)

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # transform
    trans = transforms.ToTensor()

    X_in = trans(scaled_in(m_amp_db[0, :])).unsqueeze(0).to(device, dtype=torch.float)
    
    if model_type == 'Unet':
        model = UNet(start_fm=32).to(device)
        try:
            model.load_state_dict(torch.load('./model/unet.pth'))
        except:
            st.error('Your weight are not found. Make sure the weight located in ./model')

    else:
        model = UNet_ResNet(start_fm=16).to(device)
        try:
            model.load_state_dict(torch.load('./model/unetres.pth'))
        except:
            st.error('Your weight are not found. Make sure the weight located in ./model')

    model.eval()
    X_pred = model(X_in)

    pred_amp_db = inv_scaled_ou(X_pred.detach().cpu().numpy())

    X_denoise = m_amp_db - pred_amp_db[0]

    ou_audio = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha, FRAME_LENGTH, HOP_LENGTH_FFT)

    nb_samples = ou_audio.shape[0]

    denoise_long = ou_audio.reshape(1, nb_samples*FRAME_LENGTH)*10

    sf.write(UPLOAD_FOLDER + 'out_' + filename, denoise_long[0, :], SAMPLE_RATE)

    return m_amp_db, m_pha, pred_amp_db, X_denoise 


def plot_spectrogram(stftaudio_magnitude_db, name, sample_rate=SAMPLE_RATE, hop_length_fft=HOP_LENGTH_FFT):
    """This function plots a spectrogram"""
    fig = plt.Figure(figsize=(10, 4), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(stftaudio_magnitude_db, ax=ax, x_axis='time', y_axis='log',
                             sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar(p, ax=ax)

    fig.savefig(UPLOAD_FOLDER+name, pad_inches=0)
    return 

def plot_time_serie(audio_path, name, sample_rate=SAMPLE_RATE):
    """This function plots the audio as a time serie"""
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MIN_DURATION)

    fig = plt.Figure(figsize=(10, 4), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    p = librosa.display.waveplot(audio, sr=sr, ax=ax)

    fig.savefig(UPLOAD_FOLDER+name, pad_inches=0)
    return 

def analyst_result(filename, m_amp_db, m_pha, pred_amp_db, X_denoise):  
    # noisy analyst
    plot_spectrogram(m_amp_db[0], 'noisy_spec.png')
    plot_time_serie(UPLOAD_FOLDER+filename, 'noisy_time_serie.png')

    # recreate noise + noise analyst
    pred_amp_db = pred_amp_db[0,:,:]
    plot_spectrogram(pred_amp_db[0], 'noise_spec.png')
    noise_audio = matrix_spectrogram_to_numpy_audio(pred_amp_db, m_pha, FRAME_LENGTH, HOP_LENGTH_FFT)
    # nb_samples = noise_audio.shape[0]
    # noise_long = noise_audio.reshape(1, nb_samples*FRAME_LENGTH)*10
    # sf.write(UPLOAD_FOLDER + 'noise_' + filename, noise_long[0, :], SAMPLE_RATE)
    # plot_time_serie(UPLOAD_FOLDER + 'noise_' + filename, 'noise_time_serie.png')

    # output analyst
    plot_spectrogram(X_denoise[0], 'out_spec.png')
    plot_time_serie(UPLOAD_FOLDER+'out_'+filename, 'out_time_serie.png')
    
    return