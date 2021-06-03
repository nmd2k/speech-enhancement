import requests
import numpy as np
import librosa
import streamlit as st
from utils.app_utils import *
from model.config import *

@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()


def main():
    st.title(':musical_note: Audio Noise Reduction')
    st.subheader('Remove your audio background noise using Artificial Intelligence')

    sess = load_session()

    uploaded_file = st.file_uploader("Upload your audio/video:", type=['mp3', 'mp4', 'wav'])
    
    trigger = False

    if uploaded_file != None:
        st.subheader('Input audio/video')
        trigger = True

        if uploaded_file.type == 'audio/mpeg':
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format='audio/mpeg')

        elif uploaded_file.type == 'video/mp4':
            video_bytes = uploaded_file.read()
            st.video(video_bytes)

    col1, col2, col3 = st.beta_columns([1,1,1])

    if trigger and col2.button('Start reducing'):
        pass


if __name__ == '__main__':

    st.set_page_config(
        page_title="Noise Reduction",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    
    main()
