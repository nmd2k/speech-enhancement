import os
from re import S
import requests
import numpy as np
from PIL import Image
import streamlit as st
import soundfile as sf
from utils.app_utils import model_denoising, analyst_result, play_file_uploaded, process_input_format
from model.config import *
import moviepy.editor as mp

@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()

def save_uploadedfile(uploadedfile, file_type):
    filename = uploadedfile.name

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    with open(os.path.join(UPLOAD_FOLDER, uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    
    process_input_format(filename, file_type)
    # return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

def main():
    st.title(':musical_note: Audio Noise Reduction')
    st.subheader('Remove your audio background noise using Artificial Intelligence')

    sess = load_session()

    uploaded_file = st.file_uploader("Upload your audio/video:", type=SUPPORT_FORMAT)

    # type = None
    file_type = ''
    file_name = ''
    file_format = ''

    # Play uploaded file
    if uploaded_file is not None:
        st.subheader('Input audio/video')

        file_type = uploaded_file.type
        file_name = uploaded_file.name
        file_format = file_name[-3:]

        if file_format not in SUPPORT_FORMAT:
            st.error('We are not support this format yet!')

        else:
            play_file_uploaded(uploaded_file, file_type)

    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.beta_columns([1,1,1,1,1,1,1,1,1,1,1])
    
    is_success=False

    if uploaded_file is not None and col6.button('Start reducing!'):
        # save file to backend

        save_uploadedfile(uploaded_file, file_type)

        # file_name = process_input_format(file_name, file_type)
        # file_name = file_name[:-3] + 'wav'

        m_amp_db, m_pha, pred_amp_db, X_denoise = model_denoising(file_name[:-3] + 'wav')
        analyst_result(file_name[:-3] + 'wav', m_amp_db, m_pha, pred_amp_db, X_denoise)
        is_success = True
        # except:
            # st.error('An error occurred. Please try again later')

    if is_success:
        if 'audio' in uploaded_file.type:
            out_wav = file_name[:-3] + 'wav'
            out_audio_file = open(os.path.join(UPLOAD_FOLDER, f'out_{out_wav}'), 'rb')
            out_audio_bytes = out_audio_file.read()
            st.header(':musical_note: Your processed audio/video')
            st.audio(out_audio_bytes, format='audio/wav')

        elif 'video' in uploaded_file.type:
            origin_vid =  mp.VideoFileClip(os.path.join(UPLOAD_FOLDER, file_name))
            processed_audio = mp.AudioFileClip(os.path.join(UPLOAD_FOLDER, f'out_{file_name[:-4]}.wav'))
            processed_vid = origin_vid.set_audio(processed_audio)
            processed_vid.write_videofile(UPLOAD_FOLDER + f'out_{file_name[:-4]}.mp4')

            # out_audio_file = open(os.path.join(UPLOAD_FOLDER, f'out_{file_name}'), 'rb')
            out_audio_file = open(os.path.join(UPLOAD_FOLDER, f'out_{file_name[:-4]}.mp4'), 'rb')
            out_audio_bytes = out_audio_file.read()
            st.header(':musical_note: Your processed audio/video')
            st.video(out_audio_bytes, format='video/mp4')

        st.subheader('Advanced details')
        my_expander1 = st.beta_expander('Noisy speech')
        with my_expander1:
            # st.header('Advanced details')
            st.subheader('Input detail')
            col1, col2 = st.beta_columns([1,1])
            noisy_spec          = Image.open(os.path.join(UPLOAD_FOLDER, 'noisy_spec.png'))
            noisy_time_serie    = Image.open(os.path.join(UPLOAD_FOLDER, 'noisy_time_serie.png'))
            col1.image(noisy_time_serie)
            col2.image(noisy_spec)

        my_expander2 = st.beta_expander('Noise detail')
        with my_expander2:
            st.subheader('Noise prediction')
            col1, col2 = st.beta_columns([1,1])
            noise_spec          = Image.open(os.path.join(UPLOAD_FOLDER, 'noise_spec.png'))
            # noise_time_serie    = Image.open(os.path.join(UPLOAD_FOLDER, 'noise_time_serie.png'))
            # col1.image(noise_time_serie)
            # col1.markdown("This is blank because when predicting the noise spectrogram, we don't have the true phase of that noise. \nTherefore we can not reconstruct the audio file.")
            col2.image(noise_spec)
            
        my_expander2 = st.beta_expander('Output detail')
        with my_expander2:
            st.subheader('Clean noise speech')
            col1, col2 = st.beta_columns([1,1])
            out_spec            = Image.open(os.path.join(UPLOAD_FOLDER, 'out_spec.png'))
            out_time_serie      = Image.open(os.path.join(UPLOAD_FOLDER, 'out_time_serie.png'))
            col1.image(out_time_serie)
            col2.image(out_spec)


if __name__ == '__main__':

    st.set_page_config(
        page_title="Noise Reduction",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    main()
