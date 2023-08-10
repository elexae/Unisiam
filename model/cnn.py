import torch
import torch.nn as nn

class cnn(nn.Module):
    def __init__(self, in_dim=(1,5000), emb_size=1024):
        super(cnn, self).__init__()
        self.out_dim = 256
        self.conv1 = nn.Sequential(       
            nn.Conv1d(1, 32, 5, 1, 'same'),
            nn.BatchNorm1d(32), 
            nn.ELU(),                     
            nn.Conv1d(32, 32, 5, 1, 'same'),
            nn.BatchNorm1d(32), 
            nn.ELU(),           
            nn.MaxPool1d(3, 3, 0),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(       
            nn.Conv1d(32, 64, 5, 1, 'same'),
            nn.BatchNorm1d(64), 
            nn.ReLU(),                     
            nn.Conv1d(64, 64, 5, 1, 'same'),
            nn.BatchNorm1d(64), 
            nn.ReLU(),           
            nn.MaxPool1d(3, 3, 0),
            nn.Dropout(0.1)
        )
        self.out1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(35520, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc = nn.Linear(256, emb_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.out1(x)
        if self.fc != None:
            output = self.fc(output)
        return output