"""
    Project sNNop
"""
import os
import pandas as pd
import math
import tensorflow as tf
from preprocessing import clean_data, generate_key_map
# from convNet import conv_net
from LSTM_layers import lstm
import numpy as np
import pyaudio
import wave
from pydub import AudioSegment
import simpleaudio as sa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""------------
    Data Import
------------"""
# Importing data

p = pyaudio.PyAudio()

training_data = pd.read_csv("data/8397.60171287.csv")

data = training_data.iloc[:, :-1].values

data = np.reshape(data, [-1])

max = data[0]
min = data[0]
for x in data:
    if x > max:
        max = x
    if x < min:
        min = x

print("min:", min, " max:", max)



# start playback
play_obj = sa.play_buffer(data, 1, 2, 44100)

# wait for playback to finish before exiting
play_obj.wait_done()

# data.decode()

audioSegment = AudioSegment(data=data, sample_width=2, frame_rate=8100, channels=1)
audioSegment.export("test_output.wav", format="wav")


wf = wave.open("test_output.wav", 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wf.setframerate(8100)
wf.writeframes(data)
wf.close()