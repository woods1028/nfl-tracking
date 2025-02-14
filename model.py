#%%

import torch
import torch.nn as nn
import torch.nn.functional as F1

class VideoClassifier(nn.Module):
    def __init__(self, num_classes = 6):
        super().__init__()
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Added second pooling to reduce dimensions further
        )

        # Now spatial_output_size will be smaller
        spatial_output_size = 32 * 2 * 3  # After two MaxPool operations
        total_features = spatial_output_size * 10

        self.lstm = nn.LSTM(
            input_size=total_features,
            hidden_size=64,  # Increased from 32
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x_list):
        # x_list is now a list of tensors, each with potentially different frame counts
        batch_size = len(x_list)
        
        # Process each sequence in the batch separately
        all_processed_sequences = []
        sequence_lengths = []
        
        for batch_idx in range(batch_size):
            x = x_list[batch_idx]  # Get individual sequence
            num_frames = x.size(1)  # Get actual frame count for this sequence
            sequence_lengths.append(num_frames)
            
            # Process each frame for this sequence
            processed_sequences = []
            
            for frame_idx in range(num_frames):
                frame_features = []
                
                for feature_idx in range(10):
                    frame = x[feature_idx, frame_idx, :, :].unsqueeze(0)  # Add batch dim
                    frame = frame.unsqueeze(1)  # Add channel dim
                    
                    spatial_features = self.spatial_conv(frame)
                    frame_features.append(spatial_features.view(1, -1))
                
                frame_combined = torch.cat(frame_features, dim=1)
                processed_sequences.append(frame_combined)
            
            sequence = torch.cat(processed_sequences, dim=0)  # Stack frames for this sequence
            all_processed_sequences.append(sequence)
        
        # Pack the sequences for LSTM
        packed_sequences = nn.utils.rnn.pack_sequence(all_processed_sequences)
        
        # Run through LSTM
        lstm_out, _ = self.lstm(packed_sequences)
        
        # Unpack LSTM output
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Get final output for each sequence (using actual lengths)
        final_features = torch.stack([
            lstm_out[i, length-1] 
            for i, length in enumerate(sequence_lengths)
        ])
        
        output = self.classifier(final_features)
        return output
    
#%%

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.alpha = alpha
        
    def forward(self, input, target):
        ce_loss = F1.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss