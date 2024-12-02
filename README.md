# Project Title : Predicting Emotion From Speech using CNN-LSTM model for an Speech Emotion Recognition (SER) System.

This project is a part of the AAI-501 course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

**-- Project Status: [Complete]**

**Installation**

Launch Jupyter notebook and open the G7_CNN_LSTM_HuBERT_MFCC_Augmented.ipynb file from this repository. The G7_CNN_LSTM_HuBERT_MFCC_Augmented.ipynb is the final version of the CNN-LSTM model including comparisons with the pre-trained HuBERT model from Meta.

# Required libraries to be installed including:

import math
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter

**For HuBERT Model**

Import the following:

from transformers import Wav2Vec2Processor, HubertModel, Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

** Please note, in order to utilize CUDA, the proper pytorch CUDA version needs to be installed in your environment via pip or conda install.
  
**Project Intro/Objective**

To classify 6 emotions from speech. The emotions are Anger, Sad, Neutral, Disgust, Fear, and Happy. 

**Partner(s)/Contributor(s)**

•	Donavan Trigg

•	Dean P. Simmer

•	Payal Patel

**Methods Used**

•	Classification

•	Machine Learning

•	Neural Networks

•	Deep Learning


**Technologies**

•	Python

•	Jupyter Notebook

•	pytorch


**Project Description**

Leveraing CNNs and LSTMS to train and classify emotions from speech. 

**License**

N/A

**Acknowledgments**

N/A
