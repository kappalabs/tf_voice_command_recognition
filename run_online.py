import io
import os
import time
import glob
import json
import struct
import random
import pyaudio

import numpy as np
import scipy.signal as sps

from scipy.io import wavfile
from tflite_support import metadata

from commands_execution import CommandsExecution
from commands_processor import Command, CommandsProcessor

from tflite_runtime.interpreter import Interpreter
# import tensorflow as tf

TFLITE_FILENAME = 'browserfft-speech-23012217.tflite'
SAVE_PATH = './models'
# test_dir = './dataset-test'


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
    # Convert the WAV file to a stream
    if type(wave_bytes) is str:
        wave_bytes = io.BytesIO(wave_bytes)

    # Read file
    sampling_rate, data = wavfile.read(wave_bytes)

    # Resample data
    number_of_samples = round(len(data) * float(num_samples) / sampling_rate)
    z = sps.resample(data, number_of_samples)
    z = np.asarray(z, dtype=np.float32)

    return z


def bytes_to_floats(wave_bytes, num_samples=16000):
    a = struct.unpack("%ih" % int(len(wave_bytes) / RESPEAKER_WIDTH), wave_bytes)
    a = [float(val) / pow(2, 15) for val in a]
    a = np.asarray(a, dtype=np.float32)

    return a


# Get a WAV file for inference and list of labels from the model
tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
labels = get_labels(tflite_file)
# random_audio = get_random_audio_file(test_dir)


# import tensorflow as tf
RESPEAKER_RATE = 44100
RESPEAKER_CHANNELS = 1
RESPEAKER_WIDTH = 2  # Sample width in bytes

RESPEAKER_INDEX = 1
CHUNK = RESPEAKER_RATE

p = pyaudio.PyAudio()

stream = p.open(
    rate=RESPEAKER_RATE,
    format=p.get_format_from_width(RESPEAKER_WIDTH),
    channels=RESPEAKER_CHANNELS,
    input=True,
    # input_device_index=RESPEAKER_INDEX,
)

# interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter = Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
sample_rate = get_input_sample_rate(tflite_file)
input_size = input_details[0]['shape'][1]

buffer = np.zeros((sample_rate * 2,), dtype=np.float32)
buffer_index = 0
buffer_skip = int(sample_rate / 8)

cp = CommandsProcessor()

print("Command processing is now online")

# Continuously read the input stream and classify the audio
while True:
    # Read the input stream
    data = stream.read(CHUNK)
    # Convert to float32
    data = bytes_to_floats(data, num_samples=sample_rate)
    # data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    # # Reshape to (1, 44100)
    # data = data.reshape((1, -1))
    if len(data) < input_size:
        data.resize(input_size)

    # Keep last two chunks of audio in the buffer
    if buffer_index < len(buffer):
        buffer[buffer_index:buffer_index + len(data)] = data
        buffer_index += len(data)

        # Skip the classification if buffer is not full
        continue
    else:
        buffer[:sample_rate] = buffer[sample_rate:]
        buffer[sample_rate:sample_rate + len(data)] = data

    # Run the model on the buffer with sliding window
    for i in range(0, input_size, buffer_skip):
        input_tensor = buffer[i:i + input_size]
        input_tensor = np.expand_dims(input_tensor, axis=0)
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        # Run the inference
        interpreter.invoke()
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Get the index of the highest probability
        index = np.argmax(output_data)

        if index != 0:
            command = Command(labels[index], output_data[0][index])
            print("{}: {} {}".format(time.strftime("%H:%M:%S %d.%m.%Y"), command.name, command.confidence))
            sentence = cp.process(command)
            if sentence:
                CommandsExecution.process_sentence(sentence)


stream.stop_stream()
stream.close()
p.terminate()
