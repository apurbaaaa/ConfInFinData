from flwr.client import NumPyClient, ClientApp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================
# Data Loading
# ============================
def load_data():
    data = pd.read_csv("lendingclub_partition_10.csv")

    y = data["Default"].values
    X = data.drop(columns=["Default"])

    # Encode categorical columns
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Convert to numpy float32
    X = X.astype("float32").values
    y = y.astype("float32")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test


# ============================
# Model Definition
# ============================
class LoanDefaultNN(nn.Module):
    def __init__(self, input_dim):
        super(LoanDefaultNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# ============================
# Training and Evaluation
# ============================
def train(model, train_data, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()


def test(model, test_data):
    model.eval()
    total_loss, total_accuracy, total_samples = 0.0, 0.0, 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for inputs, labels in test_data:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item() * inputs.size(0)

            predicted = (outputs.squeeze() > 0.5).float()
            acc = accuracy_score(labels.numpy(), predicted.numpy())
            total_accuracy += acc * labels.size(0)
            total_samples += labels.size(0)

    return total_loss / total_samples, total_accuracy / total_samples


# ============================
# Dataset and Model Init
# ============================
X_train, y_train, X_test, y_test = load_data()

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_dim = X_train.shape[1]
net = LoanDefaultNN(input_dim)


# ============================
# Flower Client
# ============================
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        print(f"Eval -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    return FlowerClient().to_client()


app = ClientApp(client_fn=client_fn)


if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:5006",
        client=FlowerClient().to_client(),
    )
