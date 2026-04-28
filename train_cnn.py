import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from preprocess import load_data, preprocess_data, save_pickle


class CNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * (input_dim // 4), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


class CNNClassifierWrapper:
    def __init__(self, input_dim, epochs=20, batch_size=64, lr=1e-3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNClassifier(input_dim).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X).unsqueeze(1)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            return torch.argmax(outputs, dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            return torch.softmax(outputs, dim=1).cpu().numpy()

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))


def main():
    print('\n[+] Loading dataset and preprocessing...')
    df = load_data(100000)
    X, y, scaler, features = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y)

    cnn = CNNClassifierWrapper(input_dim=X_train.shape[1], epochs=25, batch_size=128)
    cnn.fit(X_train, y_train)

    y_pred = cnn.predict(X_test)
    y_prob = cnn.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'f1': round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, y_prob) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=['Benign', 'Trojan'], zero_division=0),
    }

    save_pickle(cnn, 'cnn_model.pkl')
    cnn.save('cnn_model.pth')
    save_pickle(scaler, 'cnn_scaler.pkl')
    save_pickle(features, 'cnn_features.pkl')
    save_pickle(metrics, 'cnn_metrics.pkl')

    print('\n[+] CNN results:')
    print(f"    Accuracy : {metrics['accuracy']}%")
    print(f"    Precision: {metrics['precision']}%")
    print(f"    Recall   : {metrics['recall']}%")
    print(f"    F1 Score : {metrics['f1']}%")
    print(f"    ROC AUC  : {metrics['roc_auc']}%")
    print('\n[+] Saved CNN artifacts: cnn_model.pkl, cnn_model.pth, cnn_scaler.pkl, cnn_features.pkl, cnn_metrics.pkl')


if __name__ == '__main__':
    main()
