#!/usr/bin/env python
import os
import string
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class CharDataset(Dataset):
    def __init__(self, texts, seq_length=64):
        self.seq_length = seq_length
        
        # Create character to index mapping
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Process all text to build vocabulary
        all_text = ''.join(texts)
        chars = sorted(list(set(all_text)))
        
        for i, char in enumerate(chars):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char
        
        self.vocab_size = len(chars)
        
        # Create sequences and labels
        self.sequences = []
        self.labels = []
        
        for text in texts:
            for i in range(0, len(text) - seq_length):
                seq = text[i:i+seq_length]
                label = text[i+seq_length]
                
                # Convert to indices
                seq_idx = [self.char_to_idx[c] for c in seq]
                label_idx = self.char_to_idx[label]
                
                self.sequences.append(seq_idx)
                self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(CharLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out[:, -1, :])
        return logits


class MyModel:
    """
    Character-level language model using LSTM for multilingual text prediction.
    """
    
    def __init__(self):
        self.model = None
        self.dataset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def load_training_data(cls):
        # Load multilingual data from specified sources
        data = []
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # If no data files exist, create a simple example file
        example_file = os.path.join(data_dir, 'example.txt')
        if not os.path.exists(example_file):
            with open(example_file, 'w', encoding='utf-8') as f:
                f.write("This is a simple example text for training. It would be replaced with actual multilingual data.")
        
        # Load all text files in the data directory
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    # Simple preprocessing: normalize whitespace
                    text = ' '.join(text.split())
                    data.append(text)
        
        return data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # Create dataset
        seq_length = 64  # Sequence length hyperparameter
        self.dataset = CharDataset(data, seq_length)
        
        # Create model
        vocab_size = self.dataset.vocab_size
        self.model = CharLSTM(vocab_size)
        self.model.to(self.device)
        
        # Training parameters
        batch_size = 64        # Change batch size here
        num_epochs = 10        # Change number of epochs here
        learning_rate = 0.001  # Change learning rate here
        
        # Create DataLoader
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (sequences, labels) in enumerate(dataloader):
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    def run_pred(self, data):
        preds = []
        
        if self.model is None or self.dataset is None:
            # Fallback to random predictions if model isn't trained
            all_chars = string.ascii_letters
            for inp in data:
                top_guesses = [random.choice(all_chars) for _ in range(3)]
                preds.append(''.join(top_guesses))
            return preds
        
        seq_length = self.dataset.seq_length
        
        for inp in data:
            # If input is shorter than sequence length, pad it
            if len(inp) < seq_length:
                pad_length = seq_length - len(inp)
                inp = ' ' * pad_length + inp  # Pad with spaces
            
            # Take the last seq_length characters
            inp = inp[-seq_length:]
            
            # Convert to indices, handling unknown characters
            try:
                indices = [self.dataset.char_to_idx.get(c, 0) for c in inp]
                sequence = torch.tensor([indices], dtype=torch.long).to(self.device)
                
                # Get model prediction
                with torch.no_grad():
                    output = self.model(sequence)
                    
                # Get top 3 predictions
                _, top_indices = torch.topk(output, 3, dim=1)
                top_chars = [self.dataset.idx_to_char[idx.item()] for idx in top_indices[0]]
                
                preds.append(''.join(top_chars))
            except Exception as e:
                # Fallback to random if there's an error
                all_chars = string.ascii_letters
                top_guesses = [random.choice(all_chars) for _ in range(3)]
                preds.append(''.join(top_guesses))
        
        return preds

    def save(self, work_dir):
        # Save model and vocabulary
        if self.model is not None and self.dataset is not None:
            # Save model
            model_path = os.path.join(work_dir, 'model.pt')
            torch.save(self.model.state_dict(), model_path)
            
            # Save vocabulary and other necessary info
            vocab_path = os.path.join(work_dir, 'vocab.pt')
            torch.save({
                'char_to_idx': self.dataset.char_to_idx,
                'idx_to_char': self.dataset.idx_to_char,
                'vocab_size': self.dataset.vocab_size,
                'seq_length': self.dataset.seq_length
            }, vocab_path)
            
            print(f"Model saved to {model_path}")
            print(f"Vocabulary saved to {vocab_path}")
        else:
            # Fallback to dummy save if model is not trained
            with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
                f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        model_instance = cls()
        
        # Try to load the model and vocabulary
        model_path = os.path.join(work_dir, 'model.pt')
        vocab_path = os.path.join(work_dir, 'vocab.pt')
        
        # Determine map_location based on current device availability
        map_location = model_instance.device
        
        if os.path.exists(model_path) and os.path.exists(vocab_path):
            # Load vocabulary info
            vocab_info = torch.load(vocab_path, map_location=map_location)
            
            # Create dataset with the vocabulary
            model_instance.dataset = type('DummyDataset', (), {})
            model_instance.dataset.char_to_idx = vocab_info['char_to_idx']
            model_instance.dataset.idx_to_char = vocab_info['idx_to_char']
            model_instance.dataset.vocab_size = vocab_info['vocab_size']
            model_instance.dataset.seq_length = vocab_info['seq_length']
            
            # Create and load the model
            model_instance.model = CharLSTM(vocab_info['vocab_size'])
            # Load state_dict with map_location
            model_instance.model.load_state_dict(torch.load(model_path, map_location=map_location))
            model_instance.model.to(model_instance.device) # Move model to the correct device after loading
            model_instance.model.eval()  # Set to evaluation mode
            
            print(f"Model loaded from {model_path} to {model_instance.device}")
            print(f"Vocabulary loaded from {vocab_path}")
        else:
            # Fallback to dummy load
            with open(os.path.join(work_dir, 'model.checkpoint')) as f:
                dummy_save = f.read()
                
            print("No trained model found, using default model")
        
        return model_instance


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
