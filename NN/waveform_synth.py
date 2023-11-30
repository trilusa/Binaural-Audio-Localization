#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pysofaconventions import SOFAFile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.fftpack import fft
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from scipy.signal import convolve
import pandas as pd
import torch




# Replace this with the path to your SOFA file
#sofa_file_path = 'KEMAR/KEMAR_Knowl_EarSim_LargeEars/HRTF/HRTF/48kHz/KEMAR_Knowl_EarSim_LargeEars_FreeFieldComp_NoITD_48kHz.sofa'
sofa_file_path = 'KEMAR/KEMAR_Knowl_EarSim_LargeEars/HRTF/HRTF/48kHz/KEMAR_Knowl_EarSim_LargeEars_FreeFieldCompMinPhase_48kHz.sofa'
# sofa_file_path = 'KEMAR/KEMAR_Knowl_EarSim_LargeEars/HRTF/HRTF/48kHz/KEMAR_Knowl_EarSim_LargeEars_FreeFieldCompMinPhase_NoITD_48kHz.sofa'
# sofa_file_path = 'KEMAR/KEMAR_Knowl_EarSim_LargeEars/HRTF/HRTF/48kHz/KEMAR_Knowl_EarSim_LargeEars_FreeFieldComp_48kHz.sofa'
#sofa_file_path = 'KEMAR/KEMAR_Knowl_EarSim_LargeEars/HRTF/HRTF/48kHz/KEMAR_Knowl_EarSim_LargeEars_Raw_48kHz.sofa'

try:
    # Open the SOFA file
    sofa = SOFAFile(sofa_file_path, 'r')
    samp_rate = sofa.getSamplingRate()
    source_positions = sofa.getSourcePositionValues()
    transfer_functions = sofa.getDataIR()
    sofa.close()
except Exception as e:
    print("Error reading SOFA file:", e)
    
# Function to create sample waveforms
num_samples = 4096
duration = num_samples/samp_rate #about 25ms for 1024

t = np.linspace(0, duration, num_samples, endpoint=False)
sine_wave = np.sin(2 * np.pi * 440 * t).flatten()
white_noise = np.random.normal(0, 1, size=t.shape).flatten()
speech_like = np.sin(2 * np.pi * 440 * t) * np.sin(2 * np.pi * 20 * (t+np.random.normal(0, .001, size=t.shape))) + np.random.normal(0, .1, size=t.shape)
speech_like=speech_like.flatten()




# Synthetic HRIR data for demonstration (replace with actual HRIR data)
# # Creating a simple impulse response as an example
# synthetic_hrir = np.zeros((828, 2, 256))
# synthetic_hrir[:, 0, 0] = 1  # Left channel impulse
# synthetic_hrir[:, 1, 0] = 1  # Right channel impulse



# Create a list of base waveforms and their names
base_waveforms = [('sine_wave', sine_wave), ('white_noise', white_noise), ('speech_like', speech_like)]

# Initialize a list to store convolved waveforms and metadata
convolved_waveforms = []
metadata = []

for name, waveform in base_waveforms:
    for idx in range(transfer_functions.shape[0]):  # Looping through all HRIR positions
        hrir_left = transfer_functions[idx, 0, :]
        hrir_right = transfer_functions[idx, 1, :]
        convolved_left = convolve(waveform, hrir_left, mode='full')[:len(waveform)]
        convolved_right = convolve(waveform, hrir_right, mode='full')[:len(waveform)]
        stereo_convolved = np.stack((convolved_left, convolved_right))
        convolved_waveforms.append(stereo_convolved)
        metadata.append({'Waveform_Type': name,
                         'HRIR_Index': idx,
                         'Azimuth': source_positions[idx][0],  # Assuming the first value is azimuth
                         'Elevation': source_positions[idx][1],  # Assuming the second value is elevation
                         'Distance': source_positions[idx][2]})  # Assuming the third value is distance})

# Convert the metadata list to a DataFrame
metadata_df = pd.DataFrame(metadata)
#print(metadata_df)
# Now, convolved_waveforms contains all your convolved waveforms
# and metadata_df contains the corresponding metadata

# Process the waveforms and metadata for DataLoader
data = []  # This will store your input features
targets = []  # This will store your labels (azimuth, elevation)

for waveform, meta in zip(convolved_waveforms, metadata):
    flattened_waveform = waveform.flatten()  # Flatten the waveform
    az_el_label = [meta['Azimuth'], meta['Elevation']]  # Extract azimuth and elevation

    data.append(flattened_waveform)
    targets.append(az_el_label)

data=np.array(data)
targets=np.array(targets)

# Convert to PyTorch tensors
# data_tensor = torch.tensor(data, dtype=torch.float32)
# labels_tensor = torch.tensor(labels, dtype=torch.float32)

# torch.save(data_tensor, 'data.pt')
# torch.save(labels_tensor, 'labels.pt')

if __name__ == '__main__':
    # Plotting the waveforms
    plt.figure(figsize=(15, 5))

    plt.subplot(2, 3, 1)
    plt.plot(sine_wave)
    plt.title("Sine Wave")

    plt.subplot(2, 3, 2)
    plt.plot(white_noise)
    plt.title("White Noise")

    plt.subplot(2, 3, 3)
    plt.plot(speech_like)
    plt.title("Speech-Like Signal")
    
    # Selecting a single HRIR for demonstration (e.g., the first one)
    selected_hrir = transfer_functions[0, 0, :]
    print("HRIR Shape", selected_hrir.shape)
    print("Sine wave shape",sine_wave.shape)
    # Convolve the waveforms with the selected HRIR
    convolved_sine = convolve(sine_wave, selected_hrir, mode='full')[:len(sine_wave)]
    convolved_noise = convolve(white_noise, selected_hrir, mode='full')[:len(white_noise)]
    convolved_speech = convolve(speech_like, selected_hrir, mode='full')[:len(speech_like)]
    print("Length of waveform",len(sine_wave))
    print("Length of hrir",len(selected_hrir))
    print("Length of Connvolev",len(convolved_sine))
    plt.subplot(2, 3, 4)
    plt.plot(convolved_sine)
    plt.title("Sine Wave")
    
    plt.subplot(2, 3, 5)
    plt.plot(convolved_noise)
    plt.title("White Noise")
    
    plt.subplot(2, 3, 6)
    plt.plot(convolved_speech)
    plt.title("Speech-Like Signal")
    
    plt.tight_layout()
    plt.show()