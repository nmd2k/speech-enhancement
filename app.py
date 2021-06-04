import requests
import numpy as np
from PIL import Image
import streamlit as st
import soundfile as sf
from utils.app_utils import *
from model.config import *

@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()

def save_uploadedfile(uploadedfile):
    with open(os.path.join(UPLOAD_FOLDER, uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    # return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

def main():
    st.title(':musical_note: Audio Noise Reduction')
    st.subheader('Remove your audio background noise using Artificial Intelligence')

    sess = load_session()

    uploaded_file = st.file_uploader("Upload your audio/video:", type=['mp4', 'wav'])

    if uploaded_file is not None:
        st.subheader('Input audio/video')

        if uploaded_file.type == 'audio/wav':
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format='audio/mpeg')

        elif uploaded_file.type == 'video/mp4':
            video_bytes = uploaded_file.read()
            st.video(video_bytes)
            st.error('Not supported yet')

    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.beta_columns([1,1,1,1,1,1,1,1,1,1,1])
    
    is_success=False

    if uploaded_file is not None and col6.button('Start reducing!'):
        # save file to backend
        save_uploadedfile(uploaded_file)
        # try:
        m_amp_db, m_pha, pred_amp_db, X_denoise = model_denoising(uploaded_file.name)
        analyst_result(uploaded_file.name, m_amp_db, m_pha, pred_amp_db, X_denoise)
        is_success = True
        # except:
        #     st.error('An error occurred. Please try again later')

    if is_success:
        st.header(':musical_note: Your processed audio/video')
        
        if uploaded_file.type == 'audio/wav':
            out_audio_file = open(os.path.join(UPLOAD_FOLDER, f'out_{uploaded_file.name}'), 'rb')
            out_audio_bytes = out_audio_file.read()

            st.audio(out_audio_bytes, format=uploaded_file.type)

        elif uploaded_file.type == 'video/mp4':
            st.error('Not supported yet')
            pass
        
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
            st.subheader('Noise detection')
            col1, col2 = st.beta_columns([1,1])
            noise_spec          = Image.open(os.path.join(UPLOAD_FOLDER, 'noise_spec.png'))
            noise_time_serie    = Image.open(os.path.join(UPLOAD_FOLDER, 'noise_time_serie.png'))
            col1.image(noise_time_serie)
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
