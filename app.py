import requests
import numpy as np
import librosa
import streamlit as st
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

    is_success = False

    col1, col2, col3 = st.beta_columns([1,1,1])
    if uploaded_file is not None and col2.button('Start reducing!'):
        # save file to backend
        save_uploadedfile(uploaded_file)

        if uploaded_file.type == 'audio/wav':
            # denoising
            is_success = model_denoising(uploaded_file.name)

    if is_success:
        st.subheader(':musical_note: Your processed audio/video')
        if uploaded_file.type == 'audio/wav':
            out_audio_file = open(os.path.join(UPLOAD_FOLDER, f'out_{uploaded_file.name}'), 'rb')
            out_audio_bytes = out_audio_file.read()

            st.audio(out_audio_bytes, format=uploaded_file.type)

        elif uploaded_file.type == 'video/mp4':
            st.error('Not supported yet')
            pass

        


if __name__ == '__main__':

    st.set_page_config(
        page_title="Noise Reduction",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    
    main()
