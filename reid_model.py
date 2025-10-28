import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class SimpleReID(nn.Module):
    def __init__(self, feat_dim=128):
        super(SimpleReID, self).__init__()
        self.feat_dim = feat_dim
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(256, self.feat_dim)
        self.dropout = nn.Dropout(0.5)
    
    # === FORWARD PROPAGATION ===
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.gap(x).squeeze()
        
        x = self.fc(x)
        x = self.dropout(x)
        
        x = F.normalize(x, p=2, dim=1)
        
        return x


class ReIDExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = SimpleReID(feat_dim=128)
        self.model.eval()
        
        if torch.cuda.is_available() and device == 'cuda':
            self.model = self.model.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self._initialize_weights()
    
    # === INICIALIZACIA VAH ===
    
    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
    
    # === PREPROCESSING ===
    
    def preprocess(self, roi):
        roi_resized = cv2.resize(roi, (64, 128))
        
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        
        roi_norm = roi_rgb.astype(np.float32) / 255.0
        
        roi_chw = np.transpose(roi_norm, (2, 0, 1))
        
        roi_batch = np.expand_dims(roi_chw, axis=0)
        
        tensor = torch.from_numpy(roi_batch).float()
        
        if self.device == 'cuda':
            tensor = tensor.cuda()
        
        return tensor
    
    # === EXTRACTION FEATURES ===
    
    def extract(self, roi):
        preprocessed = self.preprocess(roi)
        
        with torch.no_grad():
            features = self.model(preprocessed)
        
        features_np = features.cpu().numpy()[0]
        
        return features_np
