import os

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export
from tensorflow_examples.lite.model_maker.core.task import model_util

try:
  from tflite_support.metadata_writers import audio_classifier as md_writer  # pylint: disable=g-import-not-at-top
  from tflite_support.metadata_writers import metadata_info as md_info  # pylint: disable=g-import-not-at-top
  from tflite_support.metadata_writers import writer_utils  # pylint: disable=g-import-not-at-top
  ENABLE_METADATA = True
except ImportError:
  ENABLE_METADATA = False

from tensorflow_examples.lite.model_maker.core.task.model_spec.audio_spec import BaseSpec, MetadataWriter
from tensorflow_examples.lite.model_maker.core.task.model_spec.audio_spec import _load_browser_fft_preprocess_model, _load_tfjs_speech_command_model
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift


@mm_export('MyFftSpec')
class MyFftSpec(BaseSpec):
  """Model good at detecting speech commands, using Browser FFT spectrum."""

  EXPECTED_WAVEFORM_LENGTH = 44032

  # Information used to populate TFLite metadata.
  _MODEL_NAME = 'AudioClassifier'
  _MODEL_DESCRIPTION = ('Identify the most prominent type in the audio clip '
                        'from a known set of categories.')

  _MODEL_VERSION = 'v1'
  _MODEL_AUTHOR = 'TensorFlow Lite Model Maker'
  _MODEL_LICENSES = ('Apache License. Version 2.0 '
                     'http://www.apache.org/licenses/LICENSE-2.0.')

  _SAMPLE_RATE = 44100
  _CHANNELS = 1

  _INPUT_NAME = 'audio_clip'
  _INPUT_DESCRIPTION = 'Input audio clip to be classified.'

  _OUTPUT_NAME = 'probability'
  _OUTPUT_DESCRIPTION = 'Scores of the labels respectively.'

  def __init__(self, model_dir=None, strategy=None):
    """Initialize a new instance for BrowserFFT spec.

    Args:
      model_dir: The location to save the model checkpoint files.
      strategy: An instance of TF distribute strategy. If none, it will use the
        default strategy (either SingleDeviceStrategy or the current scoped
        strategy.
    """
    super(MyFftSpec, self).__init__(model_dir, strategy)
    self._preprocess_model = _load_browser_fft_preprocess_model()
    self._tfjs_sc_model = _load_tfjs_speech_command_model()

    print("HERE we are")

    self.augmentations_pipeline = Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ]
    )

  @property
  def target_sample_rate(self):
    return 44100

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.float32),
      tf.TensorSpec([], dtype=tf.int32)
  ])
  def _ensure_length(self, wav, unused_label):
    return len(wav) >= self.EXPECTED_WAVEFORM_LENGTH

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.float32),
      tf.TensorSpec([], dtype=tf.int32)
  ])
  def _split(self, wav, label):
    """Split the long audio samples into multiple trunks."""
    # wav shape: (audio_samples, )
    chunks = tf.math.floordiv(len(wav), self.EXPECTED_WAVEFORM_LENGTH)
    unused = tf.math.floormod(len(wav), self.EXPECTED_WAVEFORM_LENGTH)
    # Drop unused data
    wav = wav[:len(wav) - unused]
    # Split the audio sample into multiple chunks
    wav = tf.reshape(wav, (chunks, 1, self.EXPECTED_WAVEFORM_LENGTH))

    return wav, tf.repeat(tf.expand_dims(label, 0), len(wav))

  def apply_pipeline(self, y, sr):
      shifted = self.augmentations_pipeline(y, sr)
      return shifted

  # @tf.function
  def tf_apply_pipeline(self, feature, sr, ):
      """
      Applies the augmentation pipeline to audio files
      @param y: audio data
      @param sr: sampling rate
      @return: augmented audio data
      """
      augmented_feature = tf.numpy_function(
          self.apply_pipeline, inp=[feature, sr], Tout=tf.float32, name="apply_pipeline"
      )

      return augmented_feature, sr

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[1, EXPECTED_WAVEFORM_LENGTH], dtype=tf.float32),
      tf.TensorSpec([], dtype=tf.int32)
  ])
  def _add_noise(self, embedding, label):
    print("before augmentation", embedding.shape)
    # output, _ = self.augmentations_pipeline(embedding, embedding.shape)
    output, sr = self.tf_apply_pipeline(embedding, embedding.shape)
    print("after augmentation", embedding.shape)
    # noise = tf.random.normal(
    #     embedding.shape, mean=0.0, stddev=.2, dtype=tf.dtypes.float32)
    return output, label

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[1, EXPECTED_WAVEFORM_LENGTH], dtype=tf.float32),
      tf.TensorSpec([], dtype=tf.int32)
  ])
  def _preprocess(self, x, label):
    """Preprocess the dataset to extract the spectrum."""
    # Add small Gaussian noise to the input x
    # to solve the potential "nan" problem of the preprocess_model.
    x = x + 1e-05 * tf.random.normal(x.shape)
    # x has shape (1, EXPECTED_WAVEFORM_LENGTH)
    spectrum = self._preprocess_model(x)
    # y has shape (1, embedding_len)
    spectrum = tf.squeeze(spectrum, axis=0)
    # y has shape (embedding_len,)
    return spectrum, label

  def preprocess_ds(self, ds, is_training=False, cache_fn=None):
    # del is_training

    autotune = tf.data.AUTOTUNE
    ds = ds.filter(self._ensure_length)
    ds = ds.map(self._split, num_parallel_calls=autotune).unbatch()


    print("ADDING NOISE")
    print("shape is", ds.element_spec)
    if is_training:
      ds = ds.map(self._add_noise, num_parallel_calls=autotune)
    print("END ADDED NOISE")
    print("shape is", ds.element_spec)
    # return None

    ds = ds.map(self._preprocess, num_parallel_calls=autotune)
    if cache_fn:
      ds = cache_fn(ds)

    return ds

  def create_model(self, num_classes, train_whole_model=False):
    if num_classes <= 1:
      raise ValueError(
          'AudioClassifier expects `num_classes` to be greater than 1')
    model = tf.keras.Sequential()
    for layer in self._tfjs_sc_model.layers[:-1]:
      model.add(layer)
    model.add(
        tf.keras.layers.Dense(
            name='classification_head', units=num_classes,
            activation='softmax'))
    if not train_whole_model:
      # Freeze all but the last layer of the model. The last layer will be
      # fine-tuned during transfer learning.
      for layer in model.layers[:-1]:
        layer.trainable = False
    return model

  def run_classifier(self, model, epochs, train_ds, validation_ds, **kwargs):
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    hist = model.fit(
        train_ds, validation_data=validation_ds, epochs=epochs, **kwargs)
    return hist

  def create_serving_model(self, training_model):
    """Create a model for serving."""
    combined = tf.keras.Sequential()
    combined.add(self._preprocess_model)
    combined.add(training_model)
    # Build the model.
    combined.build([None, self.EXPECTED_WAVEFORM_LENGTH])
    return combined

  def _export_metadata(self, tflite_filepath, index_to_label,
                       export_metadata_json_file):
    """Export TFLite metadata."""
    with MetadataWriter(
        tflite_filepath,
        name=self._MODEL_NAME,
        description=self._MODEL_DESCRIPTION,
        version=self._MODEL_VERSION,
        author=self._MODEL_AUTHOR,
        licenses=self._MODEL_LICENSES) as writer:
      writer.add_input(
          name=self._INPUT_NAME,
          description=self._INPUT_DESCRIPTION,
          sample_rate=self._SAMPLE_RATE,
          channels=self._CHANNELS)

      writer.add_output(
          labels=index_to_label,
          name=self._OUTPUT_NAME,
          description=self._OUTPUT_DESCRIPTION)

      json_filepath = (os.path.splitext(tflite_filepath)[0] +
                       '.json') if export_metadata_json_file else None
      writer.save(tflite_filepath, json_filepath)

  def export_tflite(self,
                    model,
                    tflite_filepath,
                    with_metadata=True,
                    export_metadata_json_file=True,
                    index_to_label=None,
                    quantization_config=None):
    """Converts the retrained model to tflite format and saves it.

    This method overrides the default `CustomModel._export_tflite` method, and
    include the pre-processing in the exported TFLite library since support
    library can't handle audio tasks yet.

    Args:
      model: An instance of the keras classification model to be exported.
      tflite_filepath: File path to save tflite model.
      with_metadata: Whether the output tflite model contains metadata.
      export_metadata_json_file: Whether to export metadata in json file. If
        True, export the metadata in the same directory as tflite model.Used
        only if `with_metadata` is True.
      index_to_label: A list that map from index to label class name.
      quantization_config: Configuration for post-training quantization.
    """
    combined = self.create_serving_model(model)

    # Sets batch size from None to 1 when converting to tflite.
    model_util.set_batch_size(model, batch_size=1)

    model_util.export_tflite(
        combined, tflite_filepath, quantization_config=quantization_config)

    # Sets batch size back to None to support retraining later.
    model_util.set_batch_size(model, batch_size=None)

    if with_metadata:
      if not ENABLE_METADATA:
        print('Writing Metadata is not support in the installed tflite-support '
              'version. Please use tflite-support >= 0.2.*')
      else:
        self._export_metadata(tflite_filepath, index_to_label,
                              export_metadata_json_file)
