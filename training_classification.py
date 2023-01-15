import os
import glob
import random
import shutil

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import wave
import scipy.signal as sps
from scipy.io import wavfile
import librosa
import soundfile as sf
from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tflite_model_maker as mm
from tflite_model_maker import audio_classifier
from tflite_model_maker.config import ExportFormat
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift
from audio_spec import MyFftSpec

print(f"TensorFlow Version: {tf.__version__}")
print(f"Model Maker Version: {mm.__version__}")

dataset_dir = './mydata'
dataset_dir_new = './mydata_augmented'
test_dir = './dataset-test'

TFLITE_FILENAME = 'browserfft-speech.tflite'
SAVE_PATH = './models'

spec = audio_classifier.BrowserFftSpec()
# spec = MyFftSpec()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

augmentations_pipeline = Compose(
    [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ]
)


def split_public_background_sounds():
    # Create a list of all the background wav files
    files = glob.glob(os.path.join('./dataset-speech/_background_noise_', '*.wav'))
    files = files + glob.glob(os.path.join('./dataset-background', '*.wav'))

    background_dir = './background_tmp'
    os.makedirs(background_dir, exist_ok=True)

    # Loop through all files and split each into several one-second wav files
    for file in files:
        filename = os.path.basename(os.path.normpath(file))
        print('Splitting', filename)
        name = os.path.splitext(filename)[0]
        rate = librosa.get_samplerate(file)
        length = round(librosa.get_duration(filename=file))
        for i in range(length - 1):
            start = i * rate
            stop = (i * rate) + rate
            data, _ = sf.read(file, start=start, stop=stop)
            sf.write(os.path.join(background_dir, name + str(i) + '.wav'), data, rate)


def split_custom_background_sounds():
    # Create a list of all the background wav files
    files = glob.glob(os.path.join('./my_background_long', '*.wav'))

    background_dir = os.path.join(dataset_dir, 'background')
    os.makedirs(background_dir, exist_ok=True)

    # Loop through all files and split each into several one-second wav files
    for file in files:
        filename = os.path.basename(os.path.normpath(file))
        print('Splitting', filename)
        name = os.path.splitext(filename)[0]
        rate = librosa.get_samplerate(file)
        length = round(librosa.get_duration(filename=file))
        for i in range(length - 1):
            start = i * rate
            stop = (i * rate) + rate
            data, _ = sf.read(file, start=start, stop=stop)
            sf.write(os.path.join(background_dir, name + str(i) + '.wav'), data, rate)


def augment_data():
    num_to_generate = 3000
    dest_framerate = 44_100

    combine_probability = 0.3
    background_max_loudness = 0.4

    dirs = glob.glob(os.path.join(dataset_dir, '*' + os.path.sep))
    background_files = glob.glob(os.path.join(dataset_dir, 'background', '*.wav'))
    for directory in dirs:
        num_to_generate_for_dir = num_to_generate
        dir_basename = directory.split(os.path.sep)[-2]

        # Get the files
        files = glob.glob(os.path.join(dataset_dir, dir_basename, '*.wav'))

        # Create the destination directory
        dest_dir = os.path.join(dataset_dir_new, dir_basename)
        os.makedirs(dest_dir, exist_ok=True)

        # Copy the original files to the new directory
        for file in files:
            shutil.copy(file, dest_dir)
        num_to_generate_for_dir -= len(files)

        # # Don't augment the background sounds
        # if directory == "background":
        #     continue

        # Start the generation of combined data
        while num_to_generate_for_dir > 0:
            # Select random background sound file
            foreground_file = np.random.choice(files, 1)[0]

            output_filename = os.path.join(dest_dir, "augmented_{}"
                                       .format(os.path.basename(foreground_file).split(".")[0]))
            print("augmenting foreground file", foreground_file)

            data = np.zeros((44100,), dtype=np.float32)
            multipliers = [1]
            files_to_load = [foreground_file]

            # Prepare also the background sound
            if np.random.random() < combine_probability:
                background_file = np.random.choice(background_files, 1)[0]
                output_filename = os.path.join(dest_dir, "augmented_{}_{}"
                                           .format(os.path.basename(background_file).split(".")[0],
                                                   os.path.basename(foreground_file).split(".")[0]))
                print(" - combining with background file", background_file)

                # Choose a random background loudness
                multiplier_background = np.random.random() * background_max_loudness
                multipliers = [1 - multiplier_background, multiplier_background]

                files_to_load = [foreground_file, background_file]

            # Load/combine the audio files
            for infile_index, infile in enumerate(files_to_load):
                sampling_rate, audio_data = wavfile.read(infile)

                # Resample data
                number_of_samples = round(len(audio_data) * float(dest_framerate) / sampling_rate)
                audio_data = sps.resample(audio_data, number_of_samples) * multipliers[infile_index]
                audio_data = np.asarray(audio_data, dtype=np.float32)

                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]

                # Combine the data
                data[:len(audio_data)] += audio_data

            # Augment the data
            data = augmentations_pipeline(data, sample_rate=dest_framerate)

            output_filename = output_filename + "_{}.wav".format(np.random.randint(0, 10e9))

            # Save the resulting audio file
            wavfile.write(output_filename, dest_framerate, data.astype(np.int16))

            num_to_generate_for_dir -= 1


def create_test_dataset():
    # Now we separate some of the files that we'll use for testing:
    test_data_ratio = 0.2
    dirs = glob.glob(os.path.join(dataset_dir, '*/'))
    for dir in dirs:
        files = glob.glob(os.path.join(dir, '*.wav'))
        test_count = round(len(files) * test_data_ratio)
        random.seed(42)
        random.shuffle(files)
        # Move test samples:
        for file in files[:test_count]:
            class_dir = os.path.basename(os.path.normpath(dir))
            os.makedirs(os.path.join(test_dir, class_dir), exist_ok=True)
            os.rename(file, os.path.join(test_dir, class_dir, os.path.basename(file)))
        print('Moved', test_count, 'images from', class_dir)


def apply_pipeline(y, sr):
    shifted = augmentations_pipeline(y, sr)
    return shifted


@tf.function
def tf_apply_pipeline(feature, sr, ):
    """
    Applies the augmentation pipeline to audio files
    @param y: audio data
    @param sr: sampling rate
    @return: augmented audio data
    """
    augmented_feature = tf.numpy_function(
        apply_pipeline, inp=[feature, sr], Tout=tf.float32, name="apply_pipeline"
    )

    return augmented_feature, sr


def augment_audio_dataset(dataset: tf.data.Dataset):
    dataset = dataset.map(tf_apply_pipeline)

    return dataset


# create_test_dataset()
augment_data()
# exit()

train_data_ratio = 0.8
train_data = audio_classifier.DataLoader.from_folder(spec, dataset_dir_new, cache=True)
train_data, validation_data = train_data.split(train_data_ratio)
test_data = audio_classifier.DataLoader.from_folder(spec, test_dir, cache=True)

# train_data = augment_audio_dataset(train_data)
# validation_data = augment_audio_dataset(validation_data)

# If your dataset has fewer than 100 samples per class,
# you might want to try a smaller batch size
batch_size = 25
epochs = 25
model = audio_classifier.create(train_data, spec, validation_data, batch_size, epochs, train_whole_model=True)

model.evaluate(test_data)


def show_confusion_matrix(confusion, test_labels):
    """Compute confusion matrix and normalize."""
    confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)
    sns.set(rc={'figure.figsize': (6, 6)})
    sns.heatmap(
        confusion_normalized, xticklabels=test_labels, yticklabels=test_labels,
        cmap='Blues', annot=True, fmt='.2f', square=True, cbar=False)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')


confusion_matrix = model.confusion_matrix(test_data)
show_confusion_matrix(confusion_matrix.numpy(), test_data.index_to_label)

print(f'Exporing the model to {SAVE_PATH}')
model.export(SAVE_PATH, tflite_filename=TFLITE_FILENAME)
model.export(SAVE_PATH, export_format=[mm.ExportFormat.SAVED_MODEL, mm.ExportFormat.LABEL])
