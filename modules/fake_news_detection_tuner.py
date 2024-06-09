import os
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from keras_tuner.engine import base_tuner
from typing import NamedTuple, Dict, Text, Any

LABEL_KEY   = 'label'
FEATURE_KEY = 'text'

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=128) -> tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""
    
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern = file_pattern,
        batch_size   = batch_size,
        features     = transform_feature_spec,
        reader       = gzip_reader_fn,
        num_epochs   = num_epochs,
        label_key    = transformed_name(LABEL_KEY)
    )

    return dataset

# Vocabulary size and number of words in a sequence.
VOCAB_SIZE      = 1000
SEQUENCE_LENGTH = 500

vectorize_layer = layers.TextVectorization(
    standardize            = 'lower_and_strip_punctuation',
    max_tokens             = VOCAB_SIZE,
    output_mode            = 'int',
    output_sequence_length = SEQUENCE_LENGTH
)

def model_builder(hp):
    """Build keras tuner model"""
    embedding_dim = hp.Int('embedding_dim', min_value=30, max_value=40, step=5)
    lstm_units    = hp.Int('lstm_units', min_value=32, max_value=96, step=16)
    dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2])
    
    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, embedding_dim, name='embedding')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    model.compile(
        loss      = tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate),
        metrics   = ['accuracy']
    )
    
    model.summary()
    return model

TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', base_tuner.BaseTuner),
    ('fit_kwargs', Dict[Text, Any]),
])

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor  = 'val_accuracy',
    mode     = 'max',
    verbose  = 1,
    patience = 2
)

def tuner_fn(fn_args: FnArgs) -> None:
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Create batches of data
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, 5)
    val_set   = input_fn(fn_args.eval_files[0],  tf_transform_output, 5)

    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)
        ]]
    )
    
    # Build the model tuner
    model_tuner = kt.Hyperband(
        hypermodel   = lambda hp: model_builder(hp),
        objective    = kt.Objective('val_accuracy', direction='max'),
        max_epochs   = 5,
        factor       = 2,
        directory    = fn_args.working_dir,
        project_name = 'fake_news_detection'
    )
    model_tuner.oracle.max_trials = 10

    return TunerFnResult(
        tuner      = model_tuner,
        fit_kwargs = {
            'callbacks'        : [early_stop_callback],
            'x'                : train_set,
            'validation_data'  : val_set,
            'steps_per_epoch'  : fn_args.train_steps,
            'validation_steps' : fn_args.eval_steps
        }
    )
     
