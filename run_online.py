import os
import glob
import json
import random
# import librosa

import numpy as np
import tensorflow as tf
import scipy.signal as sps

from scipy.io import wavfile
from tflite_support import metadata

TFLITE_FILENAME = 'browserfft-speech.tflite'
SAVE_PATH = './models'
test_dir = './dataset-test'


def get_random_audio_file(samples_dir):
    files = os.path.abspath(os.path.join(samples_dir, '*/*.wav'))
    files_list = glob.glob(files)
    # random.seed(42)
    random_audio_path = random.choice(files_list)
    return random_audio_path


def get_labels(model):
    """Returns a list of labels, extracted from the model metadata."""
    displayer = metadata.MetadataDisplayer.with_model_file(model)
    labels_file = displayer.get_packed_associated_file_list()[0]
    labels = displayer.get_associated_file_buffer(labels_file).decode()
    return [line for line in labels.split('\n')]


def get_input_sample_rate(model):
    """Returns the model's expected sample rate, from the model metadata."""
    displayer = metadata.MetadataDisplayer.with_model_file(model)
    metadata_json = json.loads(displayer.get_metadata_json())
    input_tensor_metadata = metadata_json['subgraph_metadata'][0]['input_tensor_metadata'][0]
    input_content_props = input_tensor_metadata['content']['content_properties']
    return input_content_props['sample_rate']


def wav_to_floats(wave_bytes, num_samples=16000):
    # Read file
    sampling_rate, data = wavfile.read(wave_bytes)

    # Resample data
    number_of_samples = round(len(data) * float(num_samples) / sampling_rate)
    z = sps.resample(data, number_of_samples)
    z = np.asarray(z, dtype=np.float32)

    return z


# Get a WAV file for inference and list of labels from the model
tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
labels = get_labels(tflite_file)
random_audio = get_random_audio_file(test_dir)

# Ensure the audio sample fits the model input
interpreter = tf.lite.Interpreter(tflite_file)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]
sample_rate = get_input_sample_rate(tflite_file)
# audio_data, _ = librosa.load(random_audio, sr=sample_rate)
audio_data = wav_to_floats(random_audio, num_samples=sample_rate)
if len(audio_data) < input_size:
    audio_data.resize(input_size)
audio_data = np.expand_dims(audio_data[:input_size], axis=0)

# Run inference
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], audio_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Display prediction and ground truth
top_index = np.argmax(output_data[0])
label = labels[top_index]
score = output_data[0][top_index]
print('---prediction---')
print(f'Class: {label}\nScore: {score}')
print('----truth----')
print(random_audio.split('/')[-2])
