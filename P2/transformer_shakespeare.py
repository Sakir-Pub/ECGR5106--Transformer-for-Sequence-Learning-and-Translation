import torch
import os
import glob
import random

# Transformer model definition (needed for loading)
class CharTransformerModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_heads=2, dropout=0.1):
        super(CharTransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Embedding layer
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        
        # Positional encoding (fixed, non-learned)
        self.register_buffer("positional_encoding", self._generate_positional_encoding(1000, hidden_size))
        
        # Create transformer encoder layers
        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Final linear layer for prediction
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
    def _generate_positional_encoding(self, max_len, d_model):
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        # Create embeddings
        embedded = self.embedding(x)
        
        # Add positional encoding
        embedded = embedded + self.positional_encoding[:x.size(1), :].unsqueeze(0)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(embedded)
        
        # Take the last token's output for prediction
        output = self.fc(transformer_out[:, -1, :])
        
        return output

# RNN model definition (needed for loading)
class CharRNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type='lstm', num_layers=1):
        super(CharRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        
        # Choose RNN type based on parameter
        if rnn_type.lower() == 'rnn':
            self.rnn = torch.nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif rnn_type.lower() == 'lstm':
            self.rnn = torch.nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = torch.nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
            
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.rnn_type = rnn_type.lower()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])  # Get the output of the last RNN cell
        return output

def load_model(model_path):
    """Load a saved model with all its metadata."""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract metadata
    rnn_type = checkpoint.get('rnn_type', None)
    model_type = checkpoint.get('model_type', 'RNN' if rnn_type else 'Unknown')
    sequence_length = checkpoint['sequence_length']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint.get('num_layers', 1)
    num_heads = checkpoint.get('num_heads', 2) if model_type == 'Transformer' else None
    char_to_int = checkpoint['char_to_int']
    int_to_char = checkpoint['int_to_char']
    vocabulary_size = checkpoint['vocabulary_size']
    
    # Create model based on type
    if model_type == 'Transformer':
        model = CharTransformerModel(
            input_size=vocabulary_size,
            hidden_size=hidden_size,
            output_size=vocabulary_size,
            num_layers=num_layers,
            num_heads=num_heads
        )
    else:  # RNN models
        model = CharRNNModel(
            input_size=vocabulary_size,
            hidden_size=hidden_size,
            output_size=vocabulary_size,
            rnn_type=rnn_type,
            num_layers=num_layers
        )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, {
        'model_type': model_type,
        'rnn_type': rnn_type,
        'sequence_length': sequence_length,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'char_to_int': char_to_int,
        'int_to_char': int_to_char,
        'vocabulary_size': vocabulary_size
    }

def generate_text(model, metadata, seed_text, length=100):
    """Generate text using the model, starting with seed_text."""
    char_to_int = metadata['char_to_int']
    int_to_char = metadata['int_to_char']
    sequence_length = metadata['sequence_length']
    
    # Ensure seed text is at least as long as sequence length
    if len(seed_text) < sequence_length:
        print(f"Warning: Seed text shorter than sequence length ({sequence_length}), padding with spaces.")
        seed_text = " " * (sequence_length - len(seed_text)) + seed_text
    
    # If seed text is longer, use the last 'sequence_length' characters
    current_text = seed_text[-sequence_length:]
    generated_text = current_text
    
    # Generate characters one by one
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            # Convert text to indices
            char_indices = [char_to_int[c] for c in current_text]
            
            # Create input tensor
            x = torch.tensor([char_indices], dtype=torch.long)
            
            # Get prediction
            output = model(x)
            
            # Sample from output distribution (you can adjust temperature or use argmax)
            predicted_idx = torch.argmax(output, dim=1).item()
            
            # Alternatively, sample from distribution for more diverse output
            # probs = torch.nn.functional.softmax(output / temperature, dim=1)
            # predicted_idx = torch.multinomial(probs, 1).item()
            
            # Get the predicted character
            predicted_char = int_to_char[predicted_idx]
            
            # Add to generated text
            generated_text += predicted_char
            
            # Update current text for next iteration
            current_text = current_text[1:] + predicted_char
    
    return generated_text

def main():
    # Set directories
    transformer_model_dir = 'saved_models_transformer'
    rnn_model_dir = 'saved_models_shakespear'
    output_dir = 'generated_texts'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find model files
    transformer_models = glob.glob(f"{transformer_model_dir}/*.pt")
    rnn_models = glob.glob(f"{rnn_model_dir}/*.pt")
    
    # Seed text options (from Shakespeare)
    seed_texts = [
        "To be, or not to be, that is the question:",
        "All the world's a stage, and all the men and women merely players.",
        "What's in a name? That which we call a rose by any other name would smell as sweet.",
        "Now is the winter of our discontent made glorious summer by this sun of York.",
        "The quality of mercy is not strained. It droppeth as the gentle rain from heaven."
    ]
    
    # Generate texts with transformer models
    print("\nGenerating text with Transformer models:")
    for model_path in transformer_models:
        model_name = os.path.basename(model_path).split('.')[0]
        print(f"\nLoading model: {model_name}")
        
        try:
            model, metadata = load_model(model_path)
            
            # Use a random seed text
            seed_text = random.choice(seed_texts)
            
            # Generate text
            print(f"Generating with seed: '{seed_text[:20]}...'")
            generated_text = generate_text(model, metadata, seed_text, length=200)
            
            # Save to file
            output_file = f"{output_dir}/{model_name}_generated.txt"
            with open(output_file, 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Type: Transformer (layers={metadata['num_layers']}, heads={metadata['num_heads']})\n")
                f.write(f"Sequence Length: {metadata['sequence_length']}\n")
                f.write(f"Seed Text: {seed_text}\n\n")
                f.write(f"Generated Text:\n{generated_text}\n")
            
            print(f"Generated text saved to {output_file}")
            
            # Also print a snippet
            print(f"Sample: '{generated_text[:50]}...'")
            
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
    
    # Generate texts with RNN models
    print("\nGenerating text with RNN models:")
    for model_path in rnn_models:
        model_name = os.path.basename(model_path).split('.')[0]
        if 'lstm' in model_name.lower() or 'gru' in model_name.lower():  # Filter to only use LSTM and GRU models
            print(f"\nLoading model: {model_name}")
            
            try:
                model, metadata = load_model(model_path)
                
                # Use a random seed text
                seed_text = random.choice(seed_texts)
                
                # Generate text
                print(f"Generating with seed: '{seed_text[:20]}...'")
                generated_text = generate_text(model, metadata, seed_text, length=200)
                
                # Save to file
                output_file = f"{output_dir}/{model_name}_generated.txt"
                with open(output_file, 'w') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Type: {metadata['rnn_type'].upper()}\n")
                    f.write(f"Sequence Length: {metadata['sequence_length']}\n")
                    f.write(f"Seed Text: {seed_text}\n\n")
                    f.write(f"Generated Text:\n{generated_text}\n")
                
                print(f"Generated text saved to {output_file}")
                
                # Also print a snippet
                print(f"Sample: '{generated_text[:50]}...'")
                
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
    
    print(f"\nAll generated texts saved to {output_dir}/")

if __name__ == "__main__":
    main()