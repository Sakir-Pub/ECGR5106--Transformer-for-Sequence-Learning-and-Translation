import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import requests
import math

# PART 1: Setup and data loading (similar to provided code)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def download_shakespeare_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading Shakespeare data from {url}")
    response = requests.get(url)
    text = response.text
    print(f"Downloaded data with {len(text)} characters")
    return text

text = download_shakespeare_data()

# Create character mappings
chars = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Vocabulary size: {len(chars)} unique characters")

# Custom dataset class
class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

# Function to prepare dataset with variable sequence length
def prepare_dataloaders(text, sequence_length, batch_size=128):
    # Encode the text into integers
    encoded_text = [char_to_int[ch] for ch in text]
    
    # Create sequences and targets
    sequences = []
    targets = []
    for i in range(0, len(encoded_text) - sequence_length):
        seq = encoded_text[i:i+sequence_length]
        target = encoded_text[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    # Convert lists to PyTorch tensors
    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    # Instantiate the dataset
    dataset = CharDataset(sequences, targets)
    
    # Create train and test splits
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    return train_loader, test_loader

# PART 2: Define Transformer-based model
class CharTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_heads=2, dropout=0.1):
        super(CharTransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # Positional encoding (fixed, non-learned)
        self.register_buffer("positional_encoding", self._generate_positional_encoding(1000, hidden_size))
        
        # Create transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,  # Standard practice is 4x hidden size
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Final linear layer for prediction
        self.fc = nn.Linear(hidden_size, output_size)
    
    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        # Create padding mask (not needed for this specific task since all sequences have the same length)
        # mask = torch.zeros((x.size(0), x.size(1))).bool().to(x.device)
        
        # Create embeddings
        embedded = self.embedding(x)
        
        # Add positional encoding (we only need the positions up to our sequence length)
        embedded = embedded + self.positional_encoding[:x.size(1), :].unsqueeze(0)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(embedded)
        
        # Take the last token's output for prediction (like in the RNN models)
        output = self.fc(transformer_out[:, -1, :])
        
        return output

# Training function
def train_model(model, train_loader, test_loader, epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    training_history = []
    
    # Start timing for training duration
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        for batch_sequences, batch_targets in train_loader:
            # Move data to device
            batch_sequences = batch_sequences.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(batch_sequences)
            loss = criterion(output, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / total_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_sequences, batch_targets in test_loader:
                # Move data to device
                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)
                
                # Forward pass
                output = model(batch_sequences)
                loss = criterion(output, batch_targets)
                
                # Calculate metrics
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                val_correct += (predicted == batch_targets).sum().item()
                val_total += batch_targets.size(0)
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = val_correct / val_total
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        })
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    # Calculate training duration
    training_duration = time.time() - start_time
    print(f"Training completed in {training_duration:.2f} seconds")
    
    return training_history, training_duration

# Prediction function
def predict_next_char(model, initial_str, sequence_length):
    model.eval()
    with torch.no_grad():
        # Convert string to indices
        char_indices = [char_to_int[c] for c in initial_str[-sequence_length:]]
        # Create input tensor
        initial_input = torch.tensor([char_indices], dtype=torch.long).to(device)
        # Get prediction
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return int_to_char[predicted_index]

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to plot training history
def plot_training_history(histories, model_names, metric='loss', save_path=None):
    plt.figure(figsize=(12, 8))
    
    for history, name in zip(histories, model_names):
        data = [entry[metric] for entry in history]
        epochs = [entry['epoch'] for entry in history]
        plt.plot(epochs, data, label=name)
    
    plt.title(f'Training {metric} over epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# PART 3: Train and Evaluate Models

# Define directory structure for saving results
save_dir = 'saved_models_transformer'
data_dir = 'model_data_transformer'
plots_dir = 'transformer_plots'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Hyperparameters
hidden_sizes = [128]  # We'll use 128 as in the provided RNN code
epochs = 100
batch_size = 128
learning_rate = 0.001  # Typically lower learning rate for transformers

# 1. First part: Train transformers with sequence lengths 20 and 30
sequence_lengths = [20, 30]
num_layers = 2
num_heads = 2

performance_metrics = []

for seq_len in sequence_lengths:
    print(f"\n{'='*50}")
    print(f"Training Transformer model with:")
    print(f"- Sequence length: {seq_len}")
    print(f"- Hidden size: {hidden_sizes[0]}")
    print(f"- Number of layers: {num_layers}")
    print(f"- Number of heads: {num_heads}")
    print(f"{'='*50}")
    
    # Prepare dataloaders
    train_loader, test_loader = prepare_dataloaders(text, seq_len, batch_size)
    print(f"Created data loaders with sequence length {seq_len}")
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model and move to device
    model = CharTransformerModel(
        input_size=len(chars),
        hidden_size=hidden_sizes[0],
        output_size=len(chars),
        num_layers=num_layers,
        num_heads=num_heads
    ).to(device)
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Calculate model size in MB
    model_size_mb = num_params * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter
    print(f"Approximate model size: {model_size_mb:.2f} MB")
    
    # Train model
    history, training_time = train_model(
        model, 
        train_loader, 
        test_loader, 
        epochs=epochs, 
        learning_rate=learning_rate
    )
    
    # Get final metrics
    final_train_loss = history[-1]['loss']
    final_val_loss = history[-1]['val_loss']
    final_val_accuracy = history[-1]['val_accuracy']
    
    # Track metrics
    performance_metrics.append({
        'Model Type': 'Transformer',
        'Sequence Length': seq_len,
        'Hidden Size': hidden_sizes[0],
        'Num Layers': num_layers,
        'Num Heads': num_heads,
        'Parameters': num_params,
        'Model Size (MB)': model_size_mb,
        'Training Time (s)': training_time,
        'Final Training Loss': final_train_loss,
        'Final Validation Loss': final_val_loss,
        'Final Validation Accuracy': final_val_accuracy,
        'Device': str(device)
    })
    
    # Save model
    model_filename = f"{save_dir}/transformer_seq{seq_len}_h{hidden_sizes[0]}_l{num_layers}_heads{num_heads}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.Adam(model.parameters(), lr=learning_rate).state_dict(),
        'model_type': 'Transformer',
        'sequence_length': seq_len,
        'hidden_size': hidden_sizes[0],
        'num_layers': num_layers,
        'num_heads': num_heads,
        'char_to_int': char_to_int,
        'int_to_char': int_to_char,
        'vocabulary_size': len(chars),
        'training_history': history,
        'num_parameters': num_params,
        'model_size_mb': model_size_mb,
        'training_time': training_time
    }, model_filename)
    
    print(f"Model saved to {model_filename}")
    
    # Test prediction
    test_str = text[:seq_len]  # Use the first seq_len characters from the text
    predicted_char = predict_next_char(model, test_str, seq_len)
    print(f"Test prediction with input '{test_str[:10]}...': Next character: '{predicted_char}'")
    
    # Save metrics to CSV
    model_df = pd.DataFrame([performance_metrics[-1]])
    csv_filename = f"{data_dir}/transformer_seq{seq_len}_h{hidden_sizes[0]}_l{num_layers}_heads{num_heads}_metrics.csv"
    model_df.to_csv(csv_filename, index=False)
    
    # Save training history to CSV
    history_df = pd.DataFrame(history)
    history_csv = f"{data_dir}/transformer_seq{seq_len}_h{hidden_sizes[0]}_l{num_layers}_heads{num_heads}_history.csv"
    history_df.to_csv(history_csv, index=False)

# 2. Second part: Explore different transformer architectures
# Varying layers and heads
seq_len = 30  # Use 30 for architecture exploration
layers_options = [1, 2, 4]
heads_options = [2, 4]

for num_layers in layers_options:
    for num_heads in heads_options:
        print(f"\n{'='*50}")
        print(f"Training Transformer model with:")
        print(f"- Sequence length: {seq_len}")
        print(f"- Hidden size: {hidden_sizes[0]}")
        print(f"- Number of layers: {num_layers}")
        print(f"- Number of heads: {num_heads}")
        print(f"{'='*50}")
        
        # Prepare dataloaders (reuse the same sequence length)
        train_loader, test_loader = prepare_dataloaders(text, seq_len, batch_size)
        
        # Initialize model and move to device
        model = CharTransformerModel(
            input_size=len(chars),
            hidden_size=hidden_sizes[0],
            output_size=len(chars),
            num_layers=num_layers,
            num_heads=num_heads
        ).to(device)
        
        # Count parameters
        num_params = count_parameters(model)
        print(f"Model has {num_params:,} trainable parameters")
        
        # Calculate model size in MB
        model_size_mb = num_params * 4 / (1024 * 1024)
        print(f"Approximate model size: {model_size_mb:.2f} MB")
        
        # Train model
        history, training_time = train_model(
            model, 
            train_loader, 
            test_loader, 
            epochs=epochs, 
            learning_rate=learning_rate
        )
        
        # Get final metrics
        final_train_loss = history[-1]['loss']
        final_val_loss = history[-1]['val_loss']
        final_val_accuracy = history[-1]['val_accuracy']
        
        # Track metrics
        performance_metrics.append({
            'Model Type': 'Transformer',
            'Sequence Length': seq_len,
            'Hidden Size': hidden_sizes[0],
            'Num Layers': num_layers,
            'Num Heads': num_heads,
            'Parameters': num_params,
            'Model Size (MB)': model_size_mb,
            'Training Time (s)': training_time,
            'Final Training Loss': final_train_loss,
            'Final Validation Loss': final_val_loss,
            'Final Validation Accuracy': final_val_accuracy,
            'Device': str(device)
        })
        
        # Save model
        model_filename = f"{save_dir}/transformer_seq{seq_len}_h{hidden_sizes[0]}_l{num_layers}_heads{num_heads}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.Adam(model.parameters(), lr=learning_rate).state_dict(),
            'model_type': 'Transformer',
            'sequence_length': seq_len,
            'hidden_size': hidden_sizes[0],
            'num_layers': num_layers,
            'num_heads': num_heads,
            'char_to_int': char_to_int,
            'int_to_char': int_to_char,
            'vocabulary_size': len(chars),
            'training_history': history,
            'num_parameters': num_params,
            'model_size_mb': model_size_mb,
            'training_time': training_time
        }, model_filename)
        
        print(f"Model saved to {model_filename}")
        
        # Test prediction
        test_str = text[:seq_len]
        predicted_char = predict_next_char(model, test_str, seq_len)
        print(f"Test prediction with input '{test_str[:10]}...': Next character: '{predicted_char}'")
        
        # Save metrics to CSV
        model_df = pd.DataFrame([performance_metrics[-1]])
        csv_filename = f"{data_dir}/transformer_seq{seq_len}_h{hidden_sizes[0]}_l{num_layers}_heads{num_heads}_metrics.csv"
        model_df.to_csv(csv_filename, index=False)
        
        # Save training history to CSV
        history_df = pd.DataFrame(history)
        history_csv = f"{data_dir}/transformer_seq{seq_len}_h{hidden_sizes[0]}_l{num_layers}_heads{num_heads}_history.csv"
        history_df.to_csv(history_csv, index=False)

# 3. Third part: Train with sequence length 50
seq_len = 50
num_layers = 2
num_heads = 2

print(f"\n{'='*50}")
print(f"Training Transformer model with longer sequence length:")
print(f"- Sequence length: {seq_len}")
print(f"- Hidden size: {hidden_sizes[0]}")
print(f"- Number of layers: {num_layers}")
print(f"- Number of heads: {num_heads}")
print(f"{'='*50}")

# Prepare dataloaders
train_loader, test_loader = prepare_dataloaders(text, seq_len, batch_size)
print(f"Created data loaders with sequence length {seq_len}")
print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# Initialize model and move to device
model = CharTransformerModel(
    input_size=len(chars),
    hidden_size=hidden_sizes[0],
    output_size=len(chars),
    num_layers=num_layers,
    num_heads=num_heads
).to(device)

# Count parameters
num_params = count_parameters(model)
print(f"Model has {num_params:,} trainable parameters")

# Calculate model size in MB
model_size_mb = num_params * 4 / (1024 * 1024)
print(f"Approximate model size: {model_size_mb:.2f} MB")

# Train model
history, training_time = train_model(
    model, 
    train_loader, 
    test_loader, 
    epochs=epochs, 
    learning_rate=learning_rate
)

# Get final metrics
final_train_loss = history[-1]['loss']
final_val_loss = history[-1]['val_loss']
final_val_accuracy = history[-1]['val_accuracy']

# Track metrics
performance_metrics.append({
    'Model Type': 'Transformer',
    'Sequence Length': seq_len,
    'Hidden Size': hidden_sizes[0],
    'Num Layers': num_layers,
    'Num Heads': num_heads,
    'Parameters': num_params,
    'Model Size (MB)': model_size_mb,
    'Training Time (s)': training_time,
    'Final Training Loss': final_train_loss,
    'Final Validation Loss': final_val_loss,
    'Final Validation Accuracy': final_val_accuracy,
    'Device': str(device)
})

# Save model
model_filename = f"{save_dir}/transformer_seq{seq_len}_h{hidden_sizes[0]}_l{num_layers}_heads{num_heads}.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.Adam(model.parameters(), lr=learning_rate).state_dict(),
    'model_type': 'Transformer',
    'sequence_length': seq_len,
    'hidden_size': hidden_sizes[0],
    'num_layers': num_layers,
    'num_heads': num_heads,
    'char_to_int': char_to_int,
    'int_to_char': int_to_char,
    'vocabulary_size': len(chars),
    'training_history': history,
    'num_parameters': num_params,
    'model_size_mb': model_size_mb,
    'training_time': training_time
}, model_filename)

print(f"Model saved to {model_filename}")

# Test prediction
test_str = text[:seq_len]
predicted_char = predict_next_char(model, test_str, seq_len)
print(f"Test prediction with input '{test_str[:10]}...': Next character: '{predicted_char}'")

# Save metrics to CSV
model_df = pd.DataFrame([performance_metrics[-1]])
csv_filename = f"{data_dir}/transformer_seq{seq_len}_h{hidden_sizes[0]}_l{num_layers}_heads{num_heads}_metrics.csv"
model_df.to_csv(csv_filename, index=False)

# Save training history to CSV
history_df = pd.DataFrame(history)
history_csv = f"{data_dir}/transformer_seq{seq_len}_h{hidden_sizes[0]}_l{num_layers}_heads{num_heads}_history.csv"
history_df.to_csv(history_csv, index=False)

# Create a combined metrics CSV file
combined_metrics_df = pd.DataFrame(performance_metrics)
combined_metrics_filename = f"{data_dir}/transformer_all_models_metrics.csv"
combined_metrics_df.to_csv(combined_metrics_filename, index=False)
print(f"All model metrics saved to {combined_metrics_filename}")

# Display summary of trained models
print("\n" + "="*80)
print(f"TRAINED {len(performance_metrics)} TRANSFORMER MODELS")
print("="*80)
for i, metric in enumerate(performance_metrics):
    model_type = metric['Model Type']
    seq_len = metric['Sequence Length']
    num_layers = metric['Num Layers']
    num_heads = metric['Num Heads']
    accuracy = metric['Final Validation Accuracy']
    training_time = metric['Training Time (s)']
    
    print(f"{i+1}. {model_type} (seq={seq_len}, l={num_layers}, h={num_heads}): Accuracy = {accuracy:.4f}, Time = {training_time:.2f}s")
print("="*80)