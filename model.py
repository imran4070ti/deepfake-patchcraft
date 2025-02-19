from preprocessing.patch_generator import smash_n_reconstruct
import preprocessing.filters as f
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import UnidentifiedImageError, Image
import random
import numpy as np
# Define the Feature Extraction Layer



class FeatureExtractionLayer(nn.Module):
    def __init__(self):
        super(FeatureExtractionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    

# Define the Main Model
class RichPoorTextureContrastModel(nn.Module):
    def __init__(self):
        super(RichPoorTextureContrastModel, self).__init__()
        # Feature Extraction Layers for each input
        self.feature_extraction_rich = FeatureExtractionLayer()
        self.feature_extraction_poor = FeatureExtractionLayer()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.middle_convs = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()) for _ in range(3)],
            nn.BatchNorm2d(32),
            *[nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()) for _ in range(4)],
            nn.AvgPool2d(kernel_size=2),
            *[nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()) for _ in range(2)],
            nn.AvgPool2d(kernel_size=2),
            *[nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()) for _ in range(2)]
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, rich_texture, poor_texture):
        # Feature Extraction
        rich_features = self.feature_extraction_rich(rich_texture)
        poor_features = self.feature_extraction_poor(poor_texture)
        
        # Contrast Computation
        contrast = rich_features - poor_features
        
        # Further processing
        x = F.relu(self.bn1(self.conv1(contrast)))
        x = self.middle_convs(x)
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x
    