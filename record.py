import os
import time
import wave
import pyaudio
import argparse


def get_configuration():
    default_config = {
        "respeaker_rate": 44100,
        "respeaker_channels": 1,
        "respeaker_width": 2,
        "chunk": 1024,
        "record_seconds": 2,
        "record_label": "background",
        "record_filename": "record_{}.wav".format(int(time.time())),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--respeaker_rate', type=int, default=default_config['respeaker_rate'])
    parser.add_argument('--respeaker_channels', type=int, default=default_config['respeaker_channels'])
    parser.add_argument('--respeaker_width', type=int, default=default_config['respeaker_width'])
    parser.add_argument('--chunk', type=int, default=default_config['chunk'])
    parser.add_argument('--record_seconds', type=int, default=default_config['record_seconds'])
    parser.add_argument('--record_label', type=str, default=default_config['record_label'])
    parser.add_argument('--record_filename', type=str, default=default_config['record_filename'])

    args = parser.parse_args()
    default_config.update(vars(args))

    return default_config


def record(config):
    p = pyaudio.PyAudio()

    # Prepare the stream for microphone input
    stream = p.open(
        rate=config['respeaker_rate'],
        format=p.get_format_from_width(config['respeaker_width']),
        channels=config['respeaker_channels'],
        input=True,
        #input_device_index=RESPEAKER_INDEX,
    )

    print("* recording")

    # Record for a few seconds
    frames = []
    for i in range(0, int(config['respeaker_rate'] / config['chunk'] * config['record_seconds'])):
        data = stream.read(config['chunk'])
        frames.append(data)

    print("* done recording")

    # Stop recording
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    dirpath = os.path.join(os.getcwd(), 'mydata', config['record_label'])
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    filepath = os.path.join(dirpath, config['record_filename'])
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(config['respeaker_channels'])
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(config['respeaker_width'])))
    wf.setframerate(config['respeaker_rate'])
    wf.writeframes(b''.join(frames))
    wf.close()


def main():
    default_config = get_configuration()
    record(default_config)


if __name__ == "__main__":
    main()
