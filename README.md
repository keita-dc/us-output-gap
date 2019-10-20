# US Output Gap Forecast

## Contributor
Keita Miyaki

## Goal
This project aims to develop a model to accurately forecast GDP gap of the US economy using machine learning and deep learning techniques.

## Data
Macroeconomic and financial data from FRED since kate 1970's, up/downsampled to weekly frequency for data not in weekly frequency.

## Current status
Recurrent neural network model was developed and is being tuned. It overfits. This may be the best we could do with simple RNN with limited data (only 2100 weeks of data are available for the period of time, of which 1500 were assigned to training and 600 were for test).

Fluctuation of predictions could be addressed by adding LSTM layers (instead of reducing).

## Current code

I'm using 52 timesteps (i.e. one year). Data is processed through PCA (30 top principal components are used). Train data also includes output gap at the time and its shape is (1469, 52, 31). Target is output gap of 6 month forward.

```
        regressor = Sequential()

        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
        regressor.add(Dropout(drop_out))

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(drop_out))

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(drop_out))

        regressor.add(LSTM(units=50))
        regressor.add(Dropout(drop_out))

        regressor.add(Dense(units=1))

        regressor.compile(optimizer='adam', loss='mean_squared_error')

        regressor.fit(X_train_g_rnn, 
                      y_train_g_rnn, 
                      epochs=50, 
                      batch_size=50)
```

### Trials of 10 times for different numbers of LSTM layers and drop-out rates

#### Different number of LSTM layers with drop out rate at 0.5
![RNN_LSTM_2](images/6m_gap_rnn_2_LSTM_0.5_dropout.png "LSTM 2")
![RNN_LSTM_3](images/6m_gap_rnn_3_LSTM_0.5_dropout.png "LSTM 3")
![RNN_LSTM_4](images/6m_gap_rnn_4_LSTM_0.5_dropout.png "LSTM 4")
![RNN_LSTM_5](images/6m_gap_rnn_5_LSTM_0.5_dropout.png "LSTM 5")
![RNN_LSTM_6](images/6m_gap_rnn_6_LSTM_0.5_dropout.png "LSTM 6")
![RNN_LSTM_7](images/6m_gap_rnn_7_LSTM_0.5_dropout.png "LSTM 7")
![RNN_LSTM_8](images/6m_gap_rnn_8_LSTM_0.5_dropout.png "LSTM 8")
![RNN_LSTM_9](images/6m_gap_rnn_9_LSTM_0.5_dropout.png "LSTM 9")

![LSTM_comparison](images/LSTM_layer_trial.png "comparison_LSTM")
It looks having 6 or more LSTM layers stabilizes the model predictions.

#### LSTM 7 Layers with different drop out rates
![RNN_DO_0.1](images/6m_gap_rnn_7_LSTM_0.1_dropout.png "DO_0.1")
![RNN_DO_0.3](images/6m_gap_rnn_7_LSTM_0.3_dropout.png "DO_0.3")
![RNN_DO_0.5](images/6m_gap_rnn_7_LSTM_0.5_dropout.png "DO_0.5")
![RNN_DO_0.7](images/6m_gap_rnn_7_LSTM_0.7_dropout.png "DO_0.7")

#### LSTM 7 Layers, 0.5 drop-out rate, and 100 principal components
![RNN_100_pc](images/6m_gap_rnn_7_LSTM_0.7_dropout_100_pca.png "100 pc")

![drop_out_comparison](images/drop_out_trial.png "comparison_drop_out")
Not obvious but drop-out rates around 0.3~0.5 seem better.
