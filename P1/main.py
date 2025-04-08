import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
import matplotlib.pyplot as plt
import pandas as pd

# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the sequence as a string
text = """Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text.
At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.
One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.
Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.
Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.
In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology."""

print(f"Loaded text with {len(text)} characters")

# Creating character vocabulary
chars = sorted(list(set(text)))
ix_to_char = {i: ch for i, ch in enumerate(chars)}
char_to_ix = {ch: i for i, ch in enumerate(chars)}

# Define a function to prepare dataset with variable sequence length
def prepare_dataset(text, sequence_length):
    X = []
    y = []
    for i in range(len(text) - sequence_length):
        sequence = text[i:i + sequence_length]
        label = text[i + sequence_length]
        X.append([char_to_ix[char] for char in sequence])
        y.append(char_to_ix[label])
    
    X = np.array(X)
    y = np.array(y)
    
    # Splitting the dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Converting to PyTorch tensors and moving them to the selected device
    X_train = torch.tensor(X_train, dtype=torch.long).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    
    return X_train, y_train, X_val, y_val

# Transformer Model Implementation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        if d_model % 2 != 0:  # Handle odd dimensions
            pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model//2]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to be part of the module's state
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to the embedding
        # x shape: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerWithoutCrossAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=8, num_layers=2, dropout=0.1):
        super(TransformerWithoutCrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        
        # Use a regular transformer encoder (no cross-attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, 
                                              dim_feedforward=hidden_size*4, 
                                              dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.positional_encoding(embedded)  # Add positional encoding
        
        # Pass through transformer encoder
        output = self.transformer_encoder(embedded)  # [batch_size, seq_len, embedding_dim]
        
        # Use the last token's representation for prediction
        output = self.fc(output[:, -1, :])  # [batch_size, output_size]
        return output

class TransformerWithCrossAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=8, num_layers=2, dropout=0.1):
        super(TransformerWithCrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        
        # Encoder (self-attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, 
                                              dim_feedforward=hidden_size*4, 
                                              dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder (self-attention + cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead,
                                             dim_feedforward=hidden_size*4,
                                             dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.positional_encoding(embedded)  # Add positional encoding
        
        # Pass through transformer encoder
        memory = self.transformer_encoder(embedded)  # [batch_size, seq_len, embedding_dim]
        
        # Use decoder with cross-attention to the encoder output
        # For simplicity, we use the embedded input as the target (shifted by one position)
        # This simulates the auto-regressive property of next token prediction
        tgt = embedded[:, :-1, :]  # Remove last position
        padding = torch.zeros(embedded.size(0), 1, embedded.size(2), device=embedded.device)
        tgt = torch.cat([padding, tgt], dim=1)  # Add zero padding at the beginning
        
        output = self.transformer_decoder(tgt, memory)  # [batch_size, seq_len, embedding_dim]
        
        # Use the last token's representation for prediction
        output = self.fc(output[:, -1, :])  # [batch_size, output_size]
        return output

# Modified RNN model from the original code (for comparison)
class CharRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type='lstm'):
        super(CharRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # Choose RNN type based on parameter
        if rnn_type.lower() == 'rnn':
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        elif rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
            
        self.fc = nn.Linear(hidden_size, output_size)
        self.rnn_type = rnn_type.lower()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])  # Get the output of the last RNN cell
        return output

# Training function with performance tracking
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.001, batch_size=64):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track training metrics
    training_history = []
    
    # Start timing for training duration
    start_time = time.time()
    
    # Create data loaders for batch training
    if batch_size > 0 and batch_size < len(X_train):
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        use_batches = True
    else:
        use_batches = False
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        if use_batches:
            # Batch training
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            
            # Calculate average loss for the epoch
            epoch_loss /= len(X_train)
        else:
            # Full dataset training
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            _, predicted = torch.max(val_output, 1)
            val_accuracy = (predicted == y_val).float().mean()
        
        # Store metrics
        training_history.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'val_loss': val_loss.item(),
            'val_accuracy': val_accuracy.item()
        })
        
        # Print progress every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy.item():.4f}')
    
    # Calculate training duration
    training_duration = time.time() - start_time
    print(f"Training completed in {training_duration:.2f} seconds")
    
    return training_history, training_duration

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to predict the next character
def predict_next_char(model, char_to_ix, ix_to_char, initial_str, sequence_length):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-sequence_length:]], dtype=torch.long).unsqueeze(0).to(device)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[predicted_index]

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

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    hidden_size = 128
    learning_rate = 0.001
    epochs = 100
    batch_size = 64  # Use batching for more efficient training
    
    # Model types to test
    model_types = [
        'rnn',           # Vanilla RNN
        'lstm',          # LSTM
        'gru',           # GRU
        'transformer',   # Transformer without cross-attention
        'transformer_ca' # Transformer with cross-attention
    ]
    
    # Sequence lengths to test
    sequence_lengths = [10, 20, 30]
    
    # Create directories for saving models and plots
    save_dir = 'saved_models'
    plots_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Track performance metrics for all models
    performance_metrics = []
    
    # Train and save models with different configurations
    for model_type in model_types:
        for seq_len in sequence_lengths:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()} model with sequence length {seq_len}")
            print(f"{'='*60}")
            
            # Prepare dataset with current sequence length
            X_train, y_train, X_val, y_val = prepare_dataset(text, seq_len)
            
            # Initialize model based on type
            if model_type in ['rnn', 'lstm', 'gru']:
                model = CharRNNModel(len(chars), hidden_size, len(chars), rnn_type=model_type).to(device)
            elif model_type == 'transformer':
                model = TransformerWithoutCrossAttention(
                    input_size=len(chars), 
                    hidden_size=hidden_size, 
                    output_size=len(chars),
                    nhead=4,  # Reduce number of heads for small model
                    num_layers=2
                ).to(device)
            elif model_type == 'transformer_ca':
                model = TransformerWithCrossAttention(
                    input_size=len(chars), 
                    hidden_size=hidden_size, 
                    output_size=len(chars),
                    nhead=4,  # Reduce number of heads for small model
                    num_layers=2
                ).to(device)
            
            # Count parameters and estimate model size
            num_params = count_parameters(model)
            print(f"Model has {num_params:,} trainable parameters")
            
            # Estimate model size in MB (assuming 4 bytes per parameter)
            model_size_mb = num_params * 4 / (1024 * 1024)
            print(f"Approximate model size: {model_size_mb:.2f} MB")
            
            # Train model and measure time
            history, training_time = train_model(
                model, X_train, y_train, X_val, y_val, 
                epochs=epochs, 
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            
            # Get final metrics
            final_train_loss = history[-1]['loss']
            final_val_loss = history[-1]['val_loss']
            final_val_accuracy = history[-1]['val_accuracy']
            
            # Record performance metrics
            performance_metrics.append({
                'Model Type': model_type.upper(),
                'Sequence Length': seq_len,
                'Parameters': num_params,
                'Model Size (MB)': model_size_mb,
                'Training Time (s)': training_time,
                'Final Training Loss': final_train_loss,
                'Final Validation Loss': final_val_loss,
                'Final Validation Accuracy': final_val_accuracy,
                'Device': str(device)
            })
            
            # Save model
            model_filename = f"{save_dir}/{model_type}_seq{seq_len}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.Adam(model.parameters(), lr=learning_rate).state_dict(),
                'model_type': model_type,
                'sequence_length': seq_len,
                'hidden_size': hidden_size,
                'char_to_ix': char_to_ix,
                'ix_to_char': ix_to_char,
                'vocabulary_size': len(chars),
                'training_history': history,
                'num_parameters': num_params,
                'model_size_mb': model_size_mb,
                'training_time': training_time
            }, model_filename)
            
            print(f"Model saved to {model_filename}")
            
            # Test prediction
            test_str = text[:seq_len]  # Use first seq_len characters from the text
            predicted_char = predict_next_char(model, char_to_ix, ix_to_char, test_str, seq_len)
            print(f"Test prediction with input '{test_str}': Next character: '{predicted_char}'")
    
    # Convert performance metrics to DataFrame and save to CSV
    metrics_df = pd.DataFrame(performance_metrics)
    metrics_csv_path = 'model_performance_comparison.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nPerformance metrics saved to {metrics_csv_path}")
    
    # Display performance comparison table
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON ACROSS MODELS")
    print("="*100)
    print(metrics_df.to_string())
    print("="*100)
    
    # Create visualizations for comparative analysis
    # Group metrics by model type and sequence length
    grouped_metrics = {}
    for model_type in model_types:
        for seq_len in sequence_lengths:
            model_key = f"{model_type.upper()}_seq{seq_len}"
            model_data = next((m for m in performance_metrics 
                              if m['Model Type'] == model_type.upper() 
                              and m['Sequence Length'] == seq_len), None)
            if model_data:
                grouped_metrics[model_key] = model_data
    
    # Plot training time comparison
    plt.figure(figsize=(14, 8))
    model_names = list(grouped_metrics.keys())
    train_times = [grouped_metrics[name]['Training Time (s)'] for name in model_names]
    
    plt.bar(model_names, train_times)
    plt.title('Training Time Comparison Across Models')
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/training_time_comparison.png")
    
    # Plot model size comparison
    plt.figure(figsize=(14, 8))
    model_sizes = [grouped_metrics[name]['Model Size (MB)'] for name in model_names]
    
    plt.bar(model_names, model_sizes)
    plt.title('Model Size Comparison')
    plt.xlabel('Model')
    plt.ylabel('Model Size (MB)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/model_size_comparison.png")
    
    # Plot validation accuracy comparison
    plt.figure(figsize=(14, 8))
    accuracies = [grouped_metrics[name]['Final Validation Accuracy'] for name in model_names]
    
    plt.bar(model_names, accuracies)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/validation_accuracy_comparison.png")
    
    # Collect all training histories
    all_histories = []
    all_model_names = []
    
    for model_type in model_types:
        for seq_len in sequence_lengths:
            model_filename = f"{save_dir}/{model_type}_seq{seq_len}.pt"
            if os.path.exists(model_filename):
                checkpoint = torch.load(model_filename)
                all_histories.append(checkpoint['training_history'])
                all_model_names.append(f"{model_type.upper()}_seq{seq_len}")
    
    # Plot training loss curves
    plot_training_history(
        all_histories, 
        all_model_names, 
        metric='loss', 
        save_path=f"{plots_dir}/training_loss_curves.png"
    )
    
    # Plot validation loss curves
    plot_training_history(
        all_histories, 
        all_model_names, 
        metric='val_loss', 
        save_path=f"{plots_dir}/validation_loss_curves.png"
    )
    
    # Plot validation accuracy curves
    plot_training_history(
        all_histories, 
        all_model_names, 
        metric='val_accuracy', 
        save_path=f"{plots_dir}/validation_accuracy_curves.png"
    )
    
    # Create a summary table comparing model types (averaged across sequence lengths)
    summary_by_model = metrics_df.groupby('Model Type').agg({
        'Parameters': 'mean',
        'Model Size (MB)': 'mean',
        'Training Time (s)': 'mean',
        'Final Training Loss': 'mean',
        'Final Validation Loss': 'mean',
        'Final Validation Accuracy': 'mean'
    }).reset_index()
    
    print("\n" + "="*100)
    print("SUMMARY BY MODEL TYPE (AVERAGED ACROSS SEQUENCE LENGTHS)")
    print("="*100)
    print(summary_by_model.to_string())
    print("="*100)
    
    # Create a summary table comparing sequence lengths (averaged across model types)
    summary_by_seq = metrics_df.groupby('Sequence Length').agg({
        'Parameters': 'mean',
        'Model Size (MB)': 'mean',
        'Training Time (s)': 'mean',
        'Final Training Loss': 'mean',
        'Final Validation Loss': 'mean',
        'Final Validation Accuracy': 'mean'
    }).reset_index()
    
    print("\n" + "="*100)
    print("SUMMARY BY SEQUENCE LENGTH (AVERAGED ACROSS MODEL TYPES)")
    print("="*100)
    print(summary_by_seq.to_string())
    print("="*100)
    
    print("\nAll models trained and evaluated successfully!")
    print(f"Performance visualizations saved to {plots_dir}/")
    print(f"Model checkpoints saved to {save_dir}/")
    print(f"CSV report saved to {metrics_csv_path}")
    
    # Print key findings and analysis
    best_accuracy_model = metrics_df.loc[metrics_df['Final Validation Accuracy'].idxmax()]
    fastest_model = metrics_df.loc[metrics_df['Training Time (s)'].idxmin()]
    smallest_model = metrics_df.loc[metrics_df['Model Size (MB)'].idxmin()]
    
    print("\n" + "="*100)
    print("KEY FINDINGS")
    print("="*100)
    print(f"Best accuracy: {best_accuracy_model['Model Type']} with sequence length {best_accuracy_model['Sequence Length']} - {best_accuracy_model['Final Validation Accuracy']:.4f}")
    print(f"Fastest training: {fastest_model['Model Type']} with sequence length {fastest_model['Sequence Length']} - {fastest_model['Training Time (s)']:.2f}s")
    print(f"Smallest model: {smallest_model['Model Type']} with sequence length {smallest_model['Sequence Length']} - {smallest_model['Model Size (MB)']:.2f}MB")