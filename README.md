# Fake News Detection using Machine Learning

## Project Overview

This project addresses the challenge of fake news detection, a critical issue in today's digital era. Fake news can significantly impact public perception, democratic processes, and social stability. Developing machine learning models to classify news articles as real or fake can aid in combating the spread of misinformation.

The project is run on the pipeline orchestrator Apache Beam, which allows for scalable, distributed data processing. Apache Beam's flexibility enables the implementation of complex data processing workflows, ensuring efficient handling and transformation of large datasets necessary for training and evaluating machine learning models. By leveraging Apache Beam, we can streamline the preprocessing, feature extraction, model training, and evaluation phases, thereby enhancing the overall efficiency and robustness of the fake news detection system.

## Dataset

The dataset used for this project is the [Fake News Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification). It contains news articles labeled as either real or fake, providing a basis for training and evaluating the machine learning model.

## Solution Approach

The solution leverages natural language processing (NLP) techniques and machine learning algorithms:
- Data preprocessing involves removing irrelevant features, splitting the dataset into training and validation sets (80:20 ratio), and ensuring data quality through statistical analysis and anomaly detection.
- Text data is preprocessed by transforming input features to prepare them for model input.

## Model Architecture

The model architecture is designed to effectively capture semantic meaning from text:
- Input data in string format is vectorized using an embedding layer.
- The embedded vectors undergo bidirectional LSTM layers to capture temporal patterns.
- Dropout layers mitigate overfitting, and batch normalization accelerates training.
- The output layer uses softmax activation for classification into real or fake news categories.

## Training and Evaluation

Model performance is evaluated using metrics such as:
- Accuracy: Measures the percentage of correctly predicted instances.
- Area Under the Curve (AUC): Evaluates the overall quality of the model's predictions.
- Confusion Matrix Metrics: True Positives, False Positives, True Negatives, and False Negatives quantify the model's ability to classify positive and negative classes.

## Model Performance

After evaluation, the model achieved:
- Accuracy: 85.37%
- AUC: 91.64%
- Identified 12 false positives and 12 false negatives, indicating areas for improvement to reduce prediction errors.

## Deployment Options

The fake news detection model is deployed on the Lintasarta Cloudeka Server, a Platform as a Service (PaaS) offering free deployment services.

## Web Application

Access the deployed model for inference via [Fake News Detection Model](http://103.190.215.19:8501/v1/models/cc-model/metadata).

## Monitoring

Monitoring of the machine learning model can be facilitated using Prometheus, an open-source monitoring tool. Monitoring includes reporting graph operation execution times during training or inference processes (:tensorflow:core:graph_run_time_usecs).
