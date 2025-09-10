from flwr.client import NumPyClient, ClientApp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


# ============================
# Data Loading
# ============================
def load_data(client_file="client2.csv"):
    data = pd.read_csv(client_file)

    y = data["isFraud"].astype(int).values
    X = data.drop(columns=["isFraud"])

    # Scale numeric features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to numpy float32
    X = X.astype("float32")
    y = y.astype("float32")

    # Manual 50/50 split
    split_idx = int(0.5 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train, y_train, X_test, y_test


# ============================
# Model Definition
# ============================
class FraudNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        # Xavier init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)   # logits


# ============================
# Training and Evaluation
# ============================
def train(model, train_data, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_data.dataset):.4f}")


def test(model, test_data):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, total_acc, total_auc, total_samples = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for inputs, labels in test_data:
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            acc = accuracy_score(labels.numpy(), preds.numpy())
            auc = roc_auc_score(labels.numpy(), probs.numpy())

            total_acc += acc * labels.size(0)
            total_auc += auc * labels.size(0)
            total_samples += labels.size(0)

    return total_loss / total_samples, total_acc / total_samples, total_auc / total_samples


# ============================
# Dataset and Model Init
# ============================
X_train, y_train, X_test, y_test = load_data("client2.csv")

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

input_dim = X_train.shape[1]
net = FraudNN(input_dim)


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
        loss, acc, auc = test(net, testloader)
        print(f"Eval -> Loss: {loss:.4f}, Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        return loss, len(testloader.dataset), {"accuracy": acc, "auc": auc}


def client_fn(cid: str):
    return FlowerClient().to_client()


app = ClientApp(client_fn=client_fn)


if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:5006",
        client=FlowerClient().to_client(),
    )
