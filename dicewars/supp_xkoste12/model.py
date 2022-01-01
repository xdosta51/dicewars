import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pickle

import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


INPUT_SIZE = 637
OUTPUT_SIZE = 4
MODEL_PATH = r"./model.pth"

DATA_PATH = r"./data"
TRAIN_DATA_FRACTION = 0.8

LR = 0.01
EPOCHS = 1000
BATCH_SIZE = 256

class GameDataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir=DATA_PATH):
        self.data_dir = data_dir
        self.data = list()

        self.load_data()

        self.train_size = int(TRAIN_DATA_FRACTION * len(self.data))
        self.val_size = len(self.data) - self.train_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self):
        for file in os.listdir(self.data_dir):
            if file.endswith(".pickle"):
                with open(os.path.join(DATA_PATH, file), "rb") as pkl:
                    try:
                        data = pickle.load(pkl)
                    except:
                        print("bad")
                        continue
                        
                
                self.data.extend([(np.array(state + data[1], dtype=np.dtype('float32')), \
                                   np.array(data[0], dtype=np.dtype('float32'))) for state in data[2]])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(32, OUTPUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

def train_model():
    model = NeuralNetwork().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    dataset = GameDataSet()
    train_set, val_set = torch.utils.data.random_split(dataset, [dataset.train_size, dataset.val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    min_valid_loss = np.inf

    for epoch in range(EPOCHS):
        train_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            if i % 50 == 49:
                print(f"({epoch}, {i + 1}) : loss={train_loss / 50}")
                train_loss = 0.0

        valid_loss = 0.0
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss = valid_loss + loss.item()

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("saving model")

if __name__ == "__main__":
    train_model()
