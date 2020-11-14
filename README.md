# Oil-production-prediction-using-LSTM-based-on-Keras
Oil production prediction using LSTM(Long short-term memory neural network) based on Keras

A simple python script to realize the oil production prediction using LSTM.
The time series data are transformed to standard supervisory training data which a sample contains production data of several time steps as features and one time step as labels.

In prediction period, the forecasting result of each time step is added to the end of the inputs of last time step, whose the first production data is deleted, to generate the new inputs under the current time step.  
