# Vietnamese Speech-enhancement with Wave U-net

<a href="https://wandb.ai/nmd2000/Speech-enhancement/"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg" alt="Visualize in WB"></a>

<a href="https://share.streamlit.io/manhdung20112000/speech-enhancement/main/app.py"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit"></a> 


Table of Contents
================
* [Abstract](#abstract)
* [Dataset](#dataset)
* [Training](#training)
* [Result](#result)
* [Deployment](#deployment)
* [Reference](#reference)

Abstract
========
Speech enhancement is the task of improving the intelligibility and quality of a speech signal that may have been corrupted with noise or distortion, causing loss of intelligibility or quality and compromising its effectiveness in communication.

The gif below represent the concept of speech enhancement, where the input audio has been processed to reduce the signal which corresponding to the noise.

<img src="source/denoise_ts_10classes.gif" alt="Timeserie denoising" title="Speech enhancement"/>

*Source: Vincent Belz [[1]](#1)*

This repository is our assignment for Course: Speech Processing (INT3411 20), where we attempt to use U-net [[2]](#2) for Speech Enhencement task and deploy a simple web application.
 
Dataset
=======

Audios have many different ways to be represented, going from raw time series to time-frequency decompositions. By representing with Spectrogram which consist of 2D images representing sequences of Short Time Fourier Transform (STFT) with time and frequency as axes, and brightness representing the strength of a frequency component at each time frame, the input of the model will be noisy voice spectrogram and the grouth truth will be clean voice spectrogram. Therefore, the UNet will learn how to segment the clean voice region inside noisy voice spectrogram (fig above).

![demo model](source/Unet_noisyvoice_to_noisemodel.png)

**The clean voices** were approximately 10 hours of reading Vietnamese articles by us, student of Speech Processing Course at UET. 

**The environmental noise** were gathered from ESC-50 dataset [[3]](#3). However, we only focus on 20 classes which we believe are the most relevant to daily environmental noise. These classes are: 

|                 |   |             |   |                  |   |
|-----------------|---|-------------|---|------------------|---|
| vacuum cleaner  | <img src="source/vaccum-cleaner.jpg" height="100"/>  | engine      |  <img src="source/engine.jpg" height="100"/> | keyboard typing  | <img src="source/keyboard.jpg" height="100"/> |
| fireworks       | <img src="source/firework.jpg" height="100"/>  | mouse click | <img src="source/mouse-click.png" height="100"/>  | footsteps        | <img src="source/footsteps.jpg" height="100"/>  |
| clapping        | <img src="source/clapping.jpg" height="100"/> | clock alarm | <img src="source/clock-alarm.jpg" height="100"/>  | car horn         | <img src="source/car-horn.jpg" height="100"/>  |
| door wood knock | <img src="source/knock.jpg" height="100"/>  | wind        | <img src="source/wind.jpg" height="100"/>  | drinking sipping | <img src="source/drinking-sipping.jpg" height="100"/>  |
| washing machine | <img src="source/washing-machine.jpeg" height="100"/> | rain        | <img src="source/rain.png" height="100"/>  | rooster          | <img src="source/rooster.jpg" height="100"/>  |
| snoring         | <img src="source/snoring.jpg" width="100"/> | breathing   | <img src="source/breathing.jpg" height="100"/>  | toilet flush     | <img src="source/toilet-flush.jpg" height="100"/>  |
| clock tick      | <img src="source/clock-tick.jpg" height="100"/>  | laughing    | <img src="source/laughing.jpg" height="100"/>  |                  |   |


We used public source by Vincent Belz [[1]](#1) to transform the datasets, from audios to spectrograms. Audios were sampled at 8kHz and we extracted windows slighly above 1 second. Noises have been blended to clean voices with a randomization of the noise level (between 20% and 80%). 

We publish our dataset as `Artifact` in this project worksplace at Weight&Bias (WB). We highly recommend to take a look what we've done at WB:

<a href="https://wandb.ai/nmd2000/Speech-enhancement/"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg" alt="Visualize in WB"></a>

Training
========

Result
======


Deployment
=========
To build a simple web application for demonstrate, we are using **Streamlit**, which is amazing tool for guys who don't know much about *html, css and so on*. With **Streamlit**, we can code our back-end with Python, which is very cool and easy to get started with.

Streamlit share
--------------
We have publish



To install **Streamlit** and orther dependencies, run:
```
$ pip install -r requirements.txt
```

To run app **Streamlit**, run:
```
$ streamlit run app.py
```

After that, if you see the same thing as in this video, you are go to go.
[![video-demo](source/Demo-cover.png)](https://youtu.be/kKOSrEUSsVc)



Team member
===========
Dung Nguyen Manh: 
- :octocat: [manhdung20112000](https://github.com/manhdung20112000)  
- :email: [manhdung20112000@gmail.com](mailto:manhdung20112000@gmail.com)

Nguyen Phuc Hai: 
- :octocat: [HaiNguyen2903](https://github.com/hainguyen2903) 


Reference
============
<a id="1">[1]</a> 
Vincent Belz, "Speech-enhancement". Github:https://github.com/vbelz/Speech-enhancement.

<a id="2">[2]</a> 
Jansson, A., Humphrey, E., Montecchio, N., Bittner, R., Kumar, A., & Weyde, T. (2017). Singing voice separation with deep u-net convolutional networks.

<a id="3">[3]</a> 
Karol J. Piczak. 2015. ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd ACM international conference on Multimedia (MM '15). Association for Computing Machinery, New York, NY, USA, 1015–1018. DOI:https://doi.org/10.1145/2733373.2806390

<a id="4">[4]</a> 
Grais, E. M., & Plumbley, M. D. (2017, November). Single channel audio source separation using convolutional denoising autoencoders. In 2017 IEEE global conference on signal and information processing (GlobalSIP) (pp. 1265-1269). IEEE.


