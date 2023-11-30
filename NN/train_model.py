#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:40:51 2023

@author: adrian
"""

# from test_data_gen import generate_sinusoidal_data
from azel_dataloader import AzElDataset
from torch.utils.data import DataLoaders
from sklearn.model_selection import train_test_split
from waveform_synth import data, targets


# Generate data
# num_samples = 10000  # Adjust for training
# data, targets = generate_sinusoidal_data(num_samples, 96)


# Split data and targets into training and testing sets
train_data, test_data, train_targets, test_targets = train_test_split(
    data, targets, test_size=0.2, random_state=42  # 20% for testing
)

# Then create dataset instances and DataLoaders as before
train_dataset = AzElDataset(train_data, train_targets)
test_dataset = AzElDataset(test_data, test_targets)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Assuming you have a DataLoader named 'data_loader'
print( len(train_dataset.data), len(test_dataset) )
print(train_dataset[:3][:5])
for i, (batch_inputs, batch_targets) in enumerate(train_loader):
    print(f"Batch {i}")
    print("Inputs:", batch_inputs)
    print("Targets:", batch_targets)
    # Your training or processing logic here...
