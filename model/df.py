import torch
import torch.nn as nn

class df(nn.Module):
    def __init__(self, in_dim=(1,5000), emb_size=64):
        super(df, self).__init__()
        flatten_out_dim = ((((in_dim[1]//4)//4)//4)//4)*256
        self.out_dim = emb_size
        self.conv1 = nn.Sequential(       
            nn.Conv1d(1, 32, 8, 1, 'same'),
            nn.BatchNorm1d(32), 
            nn.ELU(),                     
            nn.Conv1d(32, 32, 8, 1, 'same'),
            nn.BatchNorm1d(32), 
            nn.ELU(),           
            nn.MaxPool1d(8, 4, 2),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(       
            nn.Conv1d(32, 64, 8, 1, 'same'),
            nn.BatchNorm1d(64), 
            nn.ReLU(),                     
            nn.Conv1d(64, 64, 8, 1, 'same'),
            nn.BatchNorm1d(64), 
            nn.ReLU(),           
            nn.MaxPool1d(8, 4, 2),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(       
            nn.Conv1d(64, 128, 8, 1, 'same'),
            nn.BatchNorm1d(128), 
            nn.ReLU(),                     
            nn.Conv1d(128, 128, 8, 1, 'same'),
            nn.BatchNorm1d(128), 
            nn.ReLU(),           
            nn.MaxPool1d(8, 4, 2),
            nn.Dropout(0.1)
        )
        self.conv4 = nn.Sequential(       
            nn.Conv1d(128, 256, 8, 1, 'same'),
            nn.BatchNorm1d(256), 
            nn.ReLU(),                     
            nn.Conv1d(256, 256, 8, 1, 'same'),
            nn.BatchNorm1d(256), 
            nn.ReLU(),           
            nn.MaxPool1d(8, 4, 2),
            nn.Dropout(0.1)
        )
        self.out1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_out_dim ,emb_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.out1(x)
        if self.fc == None:
            return output
        else:
            return self.fc(output)