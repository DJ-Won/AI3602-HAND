import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from traj import geohash
from traj.model import mlpMetaEmbedding
from datetime import datetime
import random

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class CustomDataset(Dataset):
    def __init__(self, data1, labels):
        self.data1 = data1
        self.labels = labels

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return torch.tensor(self.data1[idx], dtype=torch.float32),torch.tensor(self.labels[idx], dtype=torch.float32)

def collate_fn(data):
    features = [item[0] for item in data]
    labels = [item[1] for item in data]

    features.sort(key=lambda x: x.size(0), reverse=True)

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)

    return features_padded, labels


def get_queue(t,lon,lat,model):
    df = pd.read_csv("traj/preprocess.csv")
    time = int((t.hour * 3600 + t.minute * 60))
    mask = (df['time_seconds'] >= time - 1800) & (df['time_seconds'] <= time)
    df = df[mask]
    grouped = df.groupby("ID")
    datas = []
    labels = []
    count = 0
    for name, group in grouped:

        numpy_data = group.values
        numpy_data = np.transpose(numpy_data)
        time = np.array(numpy_data[1],dtype=int)
        h = numpy_data[2]
    
        label = h[-1]
        lon, lat = geohash.decode(label)
        lon = (lon - 114)*100
        lat = (lat-22)*100
        geoh = [element.split('_') for element in h]

        geoh = np.array(geoh,dtype=int)

        trajectory_length = len(time)

        middle_index = trajectory_length 

        tmp = np.concatenate((time[:, np.newaxis], geoh), axis=1)
        datas.append(tmp[:middle_index])
        labels.append((lon,lat))
    trainset = CustomDataset(datas,labels)
    trainloader = DataLoader(trainset, batch_size=len(datas),collate_fn=collate_fn)
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu()
            for i in range(len(outputs)):
                if abs(outputs[i][0]-lon)<18 or abs(outputs[i][1]-lat)< 18:
                    count+=1
    current_datetime = datetime.now()
    time = int((current_datetime.hour * 3600 + current_datetime.minute * 60))
    random.seed(time)
    random_number = random.uniform(-150, 150)
    return abs(int(count/30)+random_number)


def main(long, lat, current_datetime):
    model = mlpMetaEmbedding(12).to(device)
    model.load_state_dict(torch.load('traj/model.pth'))
    current_datetime = datetime.now()
    return get_queue(current_datetime,long,lat,model)
    
