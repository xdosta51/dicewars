import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

INPUT_SIZE = 625
OUTPUT_SIZE = 4
MODEL_PATH = r"./model.pth"

LR = 0.01
EPOCHS = 16
BATCH_SIZE = 16

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, OUTPUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        output = F.softmax(x, dim=1)
        return output

def create_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return NeuralNetwork().to(device)

def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)

def train_model():
    model = create_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        inputs, labels = None, None

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model

if __name__ == "__main__":
    model = create_model()
    model = train_model(model)
    save_model(model)
