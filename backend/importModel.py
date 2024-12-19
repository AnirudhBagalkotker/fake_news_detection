import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

current_dir = os.getcwd()
nltk_data_dir = os.path.join(current_dir, "nltk_data")

if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path = [nltk_data_dir]
os.environ["NLTK_DATA"] = nltk_data_dir

try:
    print(f"NLTK data path: {nltk.data.path}")
    stop_words = nltk.corpus.stopwords.words("english")
    wordnet = nltk.corpus.wordnet
    nltk.word_tokenize("sample text")
    print("NLTK data is already available.")
except LookupError:
    print("Downloading NLTK data...")
    nltk.download("stopwords", download_dir=nltk_data_dir)
    nltk.download("wordnet", download_dir=nltk_data_dir)
    nltk.download("punkt", download_dir=nltk_data_dir)
    nltk.download("punkt_tab", download_dir=nltk_data_dir)
    print("NLTK data downloaded successfully.")

stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.WordNetLemmatizer()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import json

import re
import warnings

warnings.filterwarnings("ignore")

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    words = [word for word in text.split() if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# load vocab from vocab.json
# with open("vocab.json", "r") as f:
#     vocab = json.load(f)


def padArticles(data, vocab=None, max_length=100):
    if vocab is None:
        print("vocab is None")
    else:
        word2idx = vocab

    tokenizedData = [word_tokenize(file.lower()) for file in data]
    indexedData = [
        [word2idx.get(word, word2idx["UNK"]) for word in file] for file in tokenizedData
    ]
    tensorData = [torch.LongTensor(file) for file in indexedData]
    truncatedData = [tensor[:max_length] for tensor in tensorData]
    paddedData = pad_sequence(truncatedData, batch_first=True, padding_value=0)
    return paddedData


def loadCheckpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer


class LSTMModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        vocabulary_size,
        hidden_dim,
        embeddings=None,
        fine_tuning=False,
    ):
        super(LSTMModel, self).__init__()

        # Set up embeddings
        if embeddings is not None:
            # Use pre-trained embeddings if provided
            self.embeddings = nn.Embedding.from_pretrained(
                torch.FloatTensor(embeddings),
                freeze=not fine_tuning,  # Allows fine-tuning if required
                padding_idx=0,  # Use 0 as the padding index
            )
        else:
            # If no embeddings provided, create a random embedding matrix
            self.embeddings = nn.Embedding(
                num_embeddings=vocabulary_size + 1,
                embedding_dim=embedding_dim,
                padding_idx=0,
            )

        # LSTM layer: 2 layers as specified
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
            bidirectional=True,  # Adds bidirectional capability
        )

        # Fully connected layer
        self.linear = nn.Linear(
            in_features=2
            * hidden_dim,  # Since LSTM is bidirectional, double the hidden_dim
            out_features=1,  # Output layer size
        )

    def forward(self, inputs):
        emb = self.embeddings(inputs)
        lstm_out, (ht, ct) = self.lstm(emb)
        h = torch.cat((ht[-2], ht[-1]), dim=1)
        x = self.linear(h)
        return x.view(-1)


class Hybrid(nn.Module):
    def __init__(
        self,
        embedding_dim,
        vocabulary_size,
        hidden_dim,
        window_size=16,
        filter_multiplier=64,
        embeddings=None,
        fine_tuning=False,
    ):
        super(Hybrid, self).__init__()

        # Embedding Layer
        if embeddings is not None:
            self.embeddings = nn.Embedding.from_pretrained(
                torch.FloatTensor(embeddings), freeze=not fine_tuning, padding_idx=0
            )
        else:
            self.embeddings = nn.Embedding(
                num_embeddings=vocabulary_size + 1,
                embedding_dim=embedding_dim,
                padding_idx=0,
            )

        # CNN Component
        self.conv1d = nn.Conv1d(embedding_dim, filter_multiplier, window_size)
        self.dropout_cnn = nn.Dropout(0.5)

        # LSTM Component
        self.lstm = nn.LSTM(
            input_size=filter_multiplier,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Fully Connected Layer
        self.fc = nn.Linear(
            2 * hidden_dim, 1
        )  # Bidirectional LSTM doubles the hidden dimension

    def forward(self, inputs):
        # Embedding
        x = self.embeddings(inputs)

        # CNN Forward Pass
        x = x.permute(0, 2, 1)  # For Conv1d input (batch, channels, sequence_length)
        x = self.conv1d(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)  # Global Max Pooling
        x = self.dropout_cnn(x)

        # Prepare for LSTM
        x = x.unsqueeze(1).repeat(
            1, inputs.size(1), 1
        )  # Repeat CNN features for sequence length

        # LSTM Forward Pass
        lstm_out, (ht, ct) = self.lstm(x)

        # Concatenate final hidden states from both directions
        h = torch.cat((ht[-2], ht[-1]), dim=1)

        # Fully Connected Layer
        output = self.fc(h)
        return output.view(-1)
