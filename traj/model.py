import torch
import torch.nn as nn
import torch.nn.functional as F
HIDDEN_SIZE = 256

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class outputLayer(nn.Module):
    def __init__(self, input_size):
        super(outputLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        l2 = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(l2)


class mlpMetaEmbedding(nn.Module):
    def __init__(self, vocal_max):
        super(mlpMetaEmbedding, self).__init__()
        self.embed = nn.Embedding(vocal_max, 100, padding_idx=0) 
        self.conv = nn.Conv1d(100, 64, kernel_size=3, padding=1)
        N = 65
        self.out = outputLayer(N)

    def forward(self, x,t):
        embedding = self.embed(x)  # B,T,32
        embedding = embedding.permute(0, 2, 1)  # B,C,T
        l1 = F.relu(self.conv(embedding))
        l1_max = torch.max(l1, dim=-1)[0]  # B,32 over-time-maxpooling
        l1_max_norm = l1_max/torch.norm(l1_max,dim=-1,keepdim=True)
        features = l1_max_norm
        features = torch.cat([features,t],dim=1)
        return self.out(features)