import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from itertools import chain


def preprocess_labels(label_series):
    """Split comma-separated labels and fit a binarizer."""
    labels_split = label_series.str.split(r"\s*,\s*")
    binarizer = MultiLabelBinarizer()
    binarizer.fit(labels_split)
    return labels_split.tolist(), binarizer

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, label_binarizer):
        self.texts = [torch.tensor([vocab[token] for token in text.split()], dtype=torch.long) for text in texts]
        self.labels = torch.tensor(label_binarizer.transform(labels), dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def build_vocab(texts, min_freq=1):
    counter = Counter(chain.from_iterable(text.split() for text in texts))
    vocab = {word: i+2 for i, (word, c) in enumerate(counter.items()) if c >= min_freq}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in texts])
    padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded, torch.stack(labels), lengths

class LSTMMultiLabelClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        out = self.fc(hn[-1])
        return out

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y, lengths in dataloader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y, lengths in dataloader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

if __name__ == "__main__":
    BIOTECH_NEWS = "biotech_news.tsv"  # path to the downloaded dataset
    df = pd.read_csv(BIOTECH_NEWS, sep="\t")
    texts = df["text"].tolist()
    labels, label_binarizer = preprocess_labels(df["labels"])

    vocab = build_vocab(texts, min_freq=2)
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_dataset = TextDataset(train_texts, train_labels, vocab, label_binarizer)
    valid_dataset = TextDataset(valid_texts, valid_labels, vocab, label_binarizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMMultiLabelClassifier(len(vocab), 128, 128, len(label_binarizer.classes_))
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

    torch.save(model.state_dict(), "lstm_multilabel.pt")
