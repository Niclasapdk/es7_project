import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_measurement_data(base_path):
    """
    Load all measurement data from the nested folder structure
    Returns a list of dictionaries with data and metadata
    """
    data_records = []
    base_path = Path(base_path)
    
    # Load data WITHOUT interference
    no_int_path = base_path / "Clean_BPSK_reference"
    if no_int_path.exists():
        for csv_file in no_int_path.glob("*.csv"):
            df = pd.read_csv(csv_file)
            data_records.append({
                'data': df[['I', 'Q']].values,  # Extract I and Q columns as numpy array
                'time': df['Time'].values,      # Keep time separately if needed
                'has_interference': False,
                'interference_freq': None,
                'interference_gain': None,
                'file_path': str(csv_file),
                'measurement_id': csv_file.stem
            })
    
    # Load data WITH interference
    with_int_path = base_path / "BPSK_Interference"
    if with_int_path.exists():
        for freq_dir in with_int_path.iterdir():
            if freq_dir.is_dir():
                freq_name = freq_dir.name  # e.g., 'freq_1'
                freq_value = extract_frequency(freq_name)  # You'll need to implement this
                
                for gain_dir in freq_dir.iterdir():
                    if gain_dir.is_dir():
                        gain_name = gain_dir.name  # e.g., 'gain_1'
                        gain_value = extract_gain(gain_name)  # You'll need to implement this
                        
                        for csv_file in gain_dir.glob("*.csv"):
                            df = pd.read_csv(csv_file)
                            data_records.append({
                                'data': df[['I', 'Q']].values,
                                'time': df['Time'].values,
                                'has_interference': True,
                                'interference_freq': freq_value,
                                'interference_gain': gain_value,
                                'file_path': str(csv_file),
                                'measurement_id': csv_file.stem
                            })
    
    return data_records

def extract_frequency(freq_dir_name):
    """Extract frequency value from directory name"""
    # Example: if directory is "freq_100MHz" or "100MHz" or "freq_1"
    # You'll need to adapt this based on your actual naming convention
    if 'freq_' in freq_dir_name:
        return freq_dir_name.replace('freq_', '')
    return freq_dir_name

def extract_gain(gain_dir_name):
    """Extract gain value from directory name"""
    # Example: if directory is "gain_10dB" or "10dB" or "gain_1"
    if 'gain_' in gain_dir_name:
        return gain_dir_name.replace('gain_', '')
    return gain_dir_name

# Usage
base_path = "maindata/"
all_data = load_measurement_data(base_path)

print(f"Total measurements: {len(all_data)}")

# Check the distribution
has_interference = sum(1 for record in all_data if record['has_interference'])
no_interference = sum(1 for record in all_data if not record['has_interference'])

print(f"With interference: {has_interference}")
print(f"Without interference: {no_interference}")

# Check interference types distribution
from collections import Counter
freq_gain_combinations = Counter()
for record in all_data:
    if record['has_interference']:
        combo = f"freq_{record['interference_freq']}_gain_{record['interference_gain']}"
        freq_gain_combinations[combo] += 1

print("\nInterference type distribution:")
for combo, count in freq_gain_combinations.items():
    print(f"  {combo}: {count} measurements")

# Check data shapes
print(f"\nData shape for first measurement: {all_data[0]['data'].shape}")
print(f"Data type: {all_data[0]['data'].dtype}")

def prepare_rnn_regression_data(data_records, sequence_length=100, target_length=1):
    """
    Prepare I/Q data for RNN regression
    
    Parameters:
    - data_records: List of loaded measurement data
    - sequence_length: Number of timesteps in input sequence
    - target_length: Number of future timesteps to predict
    
    Returns:
    - X: Input sequences of shape (samples, sequence_length, 2) [I and Q]
    - y: Target sequences of shape (samples, target_length, 2) [future I and Q]
    """
    
    X_sequences = []
    y_targets = []
    
    for record in data_records:
        iq_data = record['data']  # Shape: (total_timesteps, 2) - [I, Q]
        
        # Create sequences for this measurement
        for i in range(0, len(iq_data) - sequence_length - target_length + 1):
            # Input sequence: current I/Q values
            input_sequence = iq_data[i:i + sequence_length]  # Shape: (sequence_length, 2)
            
            # Target sequence: future I/Q values
            target_sequence = iq_data[i + sequence_length:i + sequence_length + target_length]  # Shape: (target_length, 2)
            
            X_sequences.append(input_sequence)
            y_targets.append(target_sequence)
    
    # Convert to numpy arrays
    X = np.array(X_sequences)  # Shape: (n_samples, sequence_length, 2)
    y = np.array(y_targets)    # Shape: (n_samples, target_length, 2)
    
    print(f"Input data shape: {X.shape}")   # (samples, sequence_length, 2)
    print(f"Target data shape: {y.shape}")  # (samples, target_length, 2)
    print(f"Number of sequences: {len(X_sequences)}")
    
    return X, y

# Prepare the data for regression
sequence_length = 100  # Use 100 past timesteps to predict future
target_length = 10     # Predict next 10 timesteps

X, y = prepare_rnn_regression_data(all_data, sequence_length=sequence_length, target_length=target_length)

from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, val_size=0.2):
    """Split data into train, validation, and test sets"""
    
    # First split: separate test data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    
    # Second split: separate validation data from temp
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=42, shuffle=True
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Split the data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_rnn_regression_model(input_shape, output_length):
    """
    Create RNN model for I/Q signal regression
    
    Parameters:
    - input_shape: (sequence_length, 2) for I and Q inputs
    - output_length: number of future timesteps to predict
    """
    
    model = Sequential([
        # First LSTM layer
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        # Third LSTM layer
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers for final prediction
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        
        # Output layer: predict future I and Q values
        Dense(output_length * 2),  # *2 for both I and Q
    ])
    
    # Reshape output to (target_length, 2)
    model.add(tf.keras.layers.Reshape((output_length, 2)))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error
    )
    
    return model

# Create model
input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, 2)
output_length = y_train.shape[1]  # target_length

model = create_rnn_regression_model(input_shape, output_length)
print(model.summary())

def train_regression_model(model, X_train, y_train, X_val, y_val, epochs=100):
    """Train the RNN regression model"""
    
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6, monitor='val_loss')
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Train the model
print("Training RNN regression model...")
history = train_regression_model(model, X_train, y_train, X_val, y_val, epochs=100)
