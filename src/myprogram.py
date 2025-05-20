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
    def __init__(self, texts, seq_length=64, vocab_info=None):
        self.seq_length = seq_length

        if vocab_info:
            self.char_to_idx = vocab_info['char_to_idx']
            self.idx_to_char = vocab_info['idx_to_char']
            self.vocab_size = vocab_info['vocab_size']
        else:
            self.char_to_idx = {}
            self.idx_to_char = {}
            all_text = ''.join(texts)
            chars = sorted(list(set(all_text)))
            for i, char in enumerate(chars):
                self.char_to_idx[char] = i
                self.idx_to_char[i] = char
            self.vocab_size = len(chars)

        self.sequences = []
        self.labels = []
        for text in texts:
            current_text = text if not vocab_info else ''.join([c for c in text if c in self.char_to_idx])
            for i in range(0, len(current_text) - seq_length):
                seq = current_text[i:i+seq_length]
                label = current_text[i+seq_length]
                try:
                    seq_idx = [self.char_to_idx[c] for c in seq]
                    label_idx = self.char_to_idx[label]
                    self.sequences.append(seq_idx)
                    self.labels.append(label_idx)
                except KeyError:
                    pass

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
    def clean_gutenberg_text(cls, text):
        # Basic heuristic to remove Project Gutenberg headers/footers
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG EBOOK",
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "*END THE SMALL PRINT! FOR PUBLIC DOMAIN EBOOKS*",
        ]
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG EBOOK",
            "*** END OF THE PROJECT GUTENBERG EBOOK",
            "End of the Project Gutenberg EBook",
            "End of Project Gutenberg's",
        ]

        start_pos = 0
        for marker in start_markers:
            found_pos = text.find(marker)
            if found_pos != -1:
                start_pos = max(start_pos, found_pos + len(marker))
        
        end_pos = len(text)
        for marker in end_markers:
            found_pos = text.rfind(marker) # Search from the end
            if found_pos != -1:
                end_pos = min(end_pos, found_pos)
        
        cleaned_text = text[start_pos:end_pos].strip()
        
        # Further remove lines that are excessively long and likely metadata
        lines = cleaned_text.splitlines()
        short_lines = [line for line in lines if len(line) < 200 or ' ' in line] # Keep lines with spaces or short lines
        cleaned_text = '\n'.join(short_lines)

        return cleaned_text

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
                    # Clean Gutenberg boilerplate
                    text = cls.clean_gutenberg_text(text)
                    # Simple preprocessing: normalize whitespace
                    text = ' '.join(text.split())
                    if text: # Add text only if it's not empty after cleaning
                        data.append(text)
        
        if not data:
            data.append("This is a simple example text for training. It would be replaced with actual multilingual data.")
        
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
        seq_length = 64
        batch_size = 64
        num_epochs = 10
        learning_rate = 0.001

        checkpoint_path = os.path.join(work_dir, 'checkpoint.pt')
        start_epoch = 0
        loaded_vocab_info = None

        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            start_epoch = checkpoint['epoch'] + 1
            loaded_vocab_info = {
                'char_to_idx': checkpoint['char_to_idx'],
                'idx_to_char': checkpoint['idx_to_char'],
                'vocab_size': checkpoint['vocab_size']
            }
            seq_length = checkpoint.get('seq_length', seq_length)
            print(f"Resuming training from epoch {start_epoch}")

        self.dataset = CharDataset(data, seq_length, vocab_info=loaded_vocab_info)
        vocab_size = self.dataset.vocab_size
        self.model = CharLSTM(vocab_size)
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        if loaded_vocab_info and 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Model and optimizer states loaded.")

        if start_epoch >= num_epochs:
            print(f"Training already completed up to {start_epoch-1} epochs. Final model should be in {work_dir}.")
            if not (os.path.exists(os.path.join(work_dir, 'model.pt')) and os.path.exists(os.path.join(work_dir, 'vocab.pt'))):
                self.save(work_dir)
            return

        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (sequences, labels) in enumerate(dataloader):
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

            # Save checkpoint after each epoch
            print(f'Saving checkpoint after epoch {epoch+1}...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'char_to_idx': self.dataset.char_to_idx,
                'idx_to_char': self.dataset.idx_to_char,
                'vocab_size': self.dataset.vocab_size,
                'seq_length': self.dataset.seq_length
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

        print("Training complete. Saving final model...")
        self.save(work_dir)

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
