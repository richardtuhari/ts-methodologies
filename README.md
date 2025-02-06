# ts-methodologies
Normalization methodologies for short range time series forecasting in predictive sequential evaluation frameworks

This work aims to study the power and usability of various simple machine learning and deep learning
models that work with time series of various data sources and attempt to make predictions. In the
field of time series analysis predictive models are of utmost importance when dealing with tasks like
weather forcasting, predicting measures of energy consumption, or grasp how certain economics
pointers extrapolate in time. It is widespread to use traditional statistical methods like ARIMA or
ETS, however machine learning (ML) and deep learning (DL) models, among them simple neural
networks, e.g. LSTM, are promising alternatives. In this work, the main goal is to compare the errors
and robustness of a few models and investigate how the data preprocessing may have a significant
footprint on efficiency

Requirements:
python 3.8.10
torch - 2.3.1
tqdm - 4.66.4
numpy - 1.24.4
pandas - 2.0.3
scikit-learn - 1.3.2
matplotlib - 1.3.5
