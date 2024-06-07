import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from traj import geohash
from traj.model import mlpMetaEmbedding
from datetime import datetime

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class CustomDataset(Dataset):
    def __init__(self, data1,data2, labels):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return torch.tensor(self.data1[idx], dtype=torch.long),torch.tensor(self.data2[idx], dtype=torch.float32),torch.tensor(self.labels[idx], dtype=torch.float32)

def collate_fn(data):
    features1 = [item[0] for item in data]
    features2 = [item[1] for item in data]
    labels = [item[2] for item in data]
    sorted_indices = sorted(range(len(features1)), key=lambda i: features1[i].size(0), reverse=True)
    features1 = [features1[i] for i in sorted_indices]
    features2 = [features2[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    features1_padded = torch.nn.utils.rnn.pad_sequence(features1, batch_first=True, padding_value=0)
    return features1_padded, features2,labels


def get_queue(t,lon,lat,model):
    df = pd.read_csv("traj/preprocess.csv")
    time = int((t.hour * 3600 + t.minute * 60))
    mask = (df['time_seconds'] >= time - 1800) & (df['time_seconds'] <= time)
    df = df[mask]
    grouped = df.groupby("ID")
    word_list = df['geohash'].unique().tolist()
    word_to_index = {word: idx for idx, word in enumerate(word_list)}
    count = 0
    hs = []
    ts = []
    labels = []
    for name, group in grouped:

        numpy_data = group.values
        numpy_data = np.transpose(numpy_data)
        time = np.mean(np.array(numpy_data[1],dtype=int)/86400.0)
        h = numpy_data[2]
        label = h[-1]
        lon, lat = geohash.decode(label)
        trajectory_length = len(h)
        word_indices = [word_to_index[word] for word in h]
        word_indices_tensor = torch.LongTensor(word_indices)
        middle_index = trajectory_length
        hs.append(word_indices_tensor[:middle_index])
        ts.append(time)
        labels.append((lon,lat))

    trainset = CustomDataset(hs,ts,labels)
    trainloader = DataLoader(trainset, batch_size=len(hs), shuffle=True,collate_fn=collate_fn)
    with torch.no_grad():
        for hs,ts, labels in trainloader: 
            hs = hs.to(device)
            ts = torch.stack(ts).unsqueeze(1).to(device)
            outputs = model(hs,ts).cpu()
            for i in range(len(outputs)):
                print(outputs[i])
                if abs(outputs[i][0]-lon)<0.1 and abs(outputs[i][1]-lat)< 0.1:
                    count+=1
    return int(count)


def main(long, lat, current_datetime):
    model = mlpMetaEmbedding(2268376).to(device)
    model.load_state_dict(torch.load('traj/model.pth'))
    current_datetime = datetime.now()
    return get_queue(current_datetime,long,lat,model)
    
