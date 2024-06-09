import tensorflow as tf
import os
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

LABEL_KEY   = "label"
FEATURE_KEY = "text"

# Define the set of stopwords
stop_words = set(stopwords.words('english'))

# Renaming transformed features
def transformed_name(key):
    return key + "_xf"

# Preprocess input features into transformed features
def preprocessing_fn(inputs):
    """
    inputs:  map from feature keys to raw features
    outputs: map from feature keys to transformed features
    """

    outputs = {}

    # Convert text to lowercase
    text_lower = tf.strings.lower(inputs[FEATURE_KEY])

    # Remove stopwords
    text_without_stopwords = tf.strings.regex_replace(text_lower, '|'.join(stop_words), '')

    # Store transformed text feature
    outputs[transformed_name(FEATURE_KEY)] = text_without_stopwords

    outputs[transformed_name(LABEL_KEY)]   = tf.cast(inputs[LABEL_KEY], tf.int64)


    return outputs
