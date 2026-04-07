import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import umap

import torch


def plot_embeddings(emb, labels):
    if type(emb) == torch.Tensor:
        emb = emb.cpu().numpy()

    if emb.shape[-1] > 2:
        reducer = umap.UMAP(n_components=2)
        emb = reducer.fit_transform(emb)

    # Convert labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Convert all_zsem to numpy for plotting

    # Create a scatter plot
    f = plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb[:, 0],
                          emb[:, 1],
                          c=labels_encoded,
                          cmap='magma',
                          alpha=1,
                          s=4)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Scatter plot of all_zsem colored by classes')

    # Create a legend
    handles, labels = scatter.legend_elements()
    legend_labels = label_encoder.inverse_transform(range(len(labels)))
    plt.legend(handles,
               legend_labels,
               title="Instruments",
               loc='upper left',
               bbox_to_anchor=(1.05, 1))
    plt.show()
    return f


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# 1. Dataset
class EmbeddingDataset(Dataset):

    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# 2. Model (Linear Probe)
class LinearProbe(nn.Module):

    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


# 3. Main function
def linear_probe_torch(embeddings,
                       labels,
                       epochs=10,
                       batch_size=64,
                       lr=1e-3,
                       model=None,
                       device="cpu"):
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(embeddings,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)

    # Create Datasets and Loaders
    train_dataset = EmbeddingDataset(X_train, y_train)
    test_dataset = EmbeddingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model, Loss, Optimizer
    if model is None:

        model = LinearProbe(embedding_dim=embeddings.shape[1],
                            num_classes=len(le.classes_)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Evaluation on training set
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(predictions)
            y_true.extend(y_batch.numpy())

    acc_train = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=le.classes_)
    print(f"\nLinear probe accuracy (Training): {acc_train:.4f}")
    #print(report)

    # Evaluation on test set
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(predictions)
            y_true.extend(y_batch.numpy())

    acc_val = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=le.classes_)
    print(f"\nLinear probe accuracy (Validation): {acc_val:.4f}")
    print(report)

    return model, le, acc_train, acc_val


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def knn_classify(embeddings, labels, k=5, test_size=0.2, random_state=42):
    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state)

    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)  #, metric="cosine")
    knn.fit(X_train, y_train)

    # Predict on test set
    y_pred = knn.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    print(f"KNN (k={k}) accuracy: {acc:.4f}")
    print("Classification report:\n", report)

    return knn, le, acc
