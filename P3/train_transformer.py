import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from datetime import datetime
import random
import pandas as pd
import json

from dataset_loader import DatasetLoader
from model import init_model, count_parameters, translate_sentence
from transformer_model import init_transformer_model, translate_sentence_transformer

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def train_rnn(model, dataloader, optimizer, criterion, clip=1):
    """Training function for one epoch with RNN model."""
    model.train()
    epoch_loss = 0
    
    for batch in dataloader:
        # Get the input tensor and target tensor from batch
        src = batch['encoder_inputs'].to(device)
        trg = batch['decoder_targets'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, trg)
        
        # Reshape output and target for loss calculation
        output_dim = output.shape[-1]
        
        # Exclude the first token (<SOS>)
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        # Calculate loss
        loss = criterion(output, trg)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def train_transformer(model, dataloader, optimizer, criterion, clip=1):
    """Training function for one epoch with Transformer model."""
    model.train()
    epoch_loss = 0
    
    for batch in dataloader:
        # Get the input tensor and target tensor from batch
        src = batch['encoder_inputs'].to(device)
        trg = batch['decoder_targets'].to(device)
        
        optimizer.zero_grad()
        
        # Shift trg for teacher forcing (input starts with <SOS>, target with actual tokens)
        trg_input = trg[:, :-1]  # remove last token (should be <EOS>)
        trg_target = trg[:, 1:]  # remove first token (<SOS>)
        
        # Forward pass
        output = model(src, trg_input)
        
        # Reshape output and target for loss calculation
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        trg_target = trg_target.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, trg_target)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate_rnn(model, dataloader, criterion, pad_idx):
    """Evaluation function for RNN model."""
    model.eval()
    epoch_loss = 0
    correct_tokens = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['encoder_inputs'].to(device)
            trg = batch['decoder_targets'].to(device)
            
            # Forward pass with teacher forcing disabled
            output = model(src, trg, teacher_forcing_ratio=0.0)
            
            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            
            # Exclude the first token (<SOS>)
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
            # Calculate accuracy (exclude padding tokens)
            non_pad_mask = trg != pad_idx
            pred = output.argmax(dim=1)
            correct = (pred == trg) * non_pad_mask
            correct_tokens += correct.sum().item()
            total_tokens += non_pad_mask.sum().item()
    
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return epoch_loss / len(dataloader), accuracy


def evaluate_transformer(model, dataloader, criterion, pad_idx):
    """Evaluation function for Transformer model."""
    model.eval()
    epoch_loss = 0
    correct_tokens = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['encoder_inputs'].to(device)
            trg = batch['decoder_targets'].to(device)
            
            # Shift trg for teacher forcing
            trg_input = trg[:, :-1]  # remove last token (should be <EOS>)
            trg_target = trg[:, 1:]  # remove first token (<SOS>)
            
            # Forward pass
            output = model(src, trg_input)
            
            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg_target = trg_target.reshape(-1)
            
            # Calculate loss
            loss = criterion(output, trg_target)
            epoch_loss += loss.item()
            
            # Calculate accuracy (exclude padding tokens)
            non_pad_mask = trg_target != pad_idx
            pred = output.argmax(dim=1)
            correct = (pred == trg_target) * non_pad_mask
            correct_tokens += correct.sum().item()
            total_tokens += non_pad_mask.sum().item()
    
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return epoch_loss / len(dataloader), accuracy


def print_example_translations(model, model_type, data_loader, english_vocab, french_vocab, device, num_examples=5):
    """Print example translations from the original dataset."""
    print(f"\n----- Example Translations ({model_type}) -----")
    
    if hasattr(data_loader, 'english_sentences') and hasattr(data_loader, 'french_sentences'):
        # Get random indices
        indices = random.sample(range(len(data_loader.english_sentences)), min(num_examples, len(data_loader.english_sentences)))
        
        for idx in indices:
            eng_sentence = data_loader.english_sentences[idx]
            actual_fr_sentence = data_loader.french_sentences[idx]
            
            # Get model translation
            if model_type == 'rnn':
                translated_sentence = translate_sentence(model, eng_sentence, english_vocab, french_vocab, device)
            else:  # transformer
                translated_sentence = translate_sentence_transformer(model, eng_sentence, english_vocab, french_vocab, device)
            
            print(f"English: {eng_sentence}")
            print(f"Actual French: {actual_fr_sentence}")
            print(f"Predicted French: {translated_sentence}")
            print("-" * 50)
    else:
        print("No original sentences available for translation examples.")


def save_training_metrics_csv(history, save_dir="metrics", model_type="transformer", config=""):
    """Save training metrics to a CSV file."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create DataFrame from history dictionary
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_accuracy': history['val_accuracy']
    })
    
    # Save to CSV
    csv_path = os.path.join(save_dir, f'{model_type}{config}_training_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Training metrics saved to {csv_path}")


def save_combined_plot(history, save_dir="plots", model_type="transformer", config=""):
    """Create and save a combined plot with training/validation loss and validation accuracy."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot train and validation loss in first subplot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot validation accuracy in second subplot
    ax2.plot(history['val_accuracy'], label='Val Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.0)  # Set y-axis limits for accuracy
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_type.capitalize()}{config} Model')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}{config}_combined_training_metrics.png'))
    plt.close()


def save_final_plot(history, save_dir="plots", model_type="transformer", config=""):
    """Create and save a final plot with all metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Combined plot with all metrics
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    
    # Create a secondary y-axis for accuracy
    ax2 = plt.gca().twinx()
    ax2.plot(history['val_accuracy'], label='Val Accuracy', color='green', linestyle='--')
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Accuracy')
    
    # Add labels and legend
    plt.title(f'{model_type.capitalize()}{config} Model - Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Create combined legend
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}{config}_final_training_metrics.png'))
    plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a Transformer or RNN model for English to French translation')
    
    parser.add_argument('--data_path', type=str, default="./english_french_phrases.csv",
                        help='Path to the dataset CSV file')
    parser.add_argument('--model_type', type=str, choices=['transformer', 'rnn', 'rnn_attention', 'all'], default='transformer',
                        help='Type of model to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    # Transformer-specific arguments
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension for the model')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads for transformer')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in the transformer encoder/decoder')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                        help='Dimension of feedforward network in transformer')
    
    # RNN-specific arguments
    parser.add_argument('--encoder_hidden_dim', type=int, default=256,
                        help='Hidden dimension for the encoder (RNN only)')
    parser.add_argument('--decoder_hidden_dim', type=int, default=256,
                        help='Hidden dimension for the decoder (RNN only)')
    parser.add_argument('--n_layers_rnn', type=int, default=2,
                        help='Number of layers in the encoder and decoder (RNN only)')
    
    # Common arguments
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--plot_dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--metrics_dir', type=str, default='metrics',
                        help='Directory to save metrics CSV')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--augment_ratio', type=float, default=0.5,
                        help='Ratio of data to augment (0 to disable)')
    parser.add_argument('--num_examples', type=int, default=5,
                        help='Number of example translations to print after each best epoch')
    parser.add_argument('--compare_all_configs', action='store_true',
                        help='Compare all configurations of transformer as specified in homework')
    
    return parser.parse_args()


def train_model(model, data_loader, model_type, args, config_suffix=""):
    """Train a model and save results."""
    # Get special tokens
    special_tokens = data_loader.get_special_tokens()
    pad_idx = special_tokens['french']['pad']
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # Initialize variables for tracking training
    best_val_accuracy = 0.0
    
    # For tracking metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    print(f"Starting training for {args.n_epochs} epochs...")
    start_time = datetime.now()
    
    for epoch in range(args.n_epochs):
        # Train the model
        if model_type == 'transformer':
            train_loss = train_transformer(model, data_loader.train_loader, optimizer, criterion, args.clip)
            val_loss, val_accuracy = evaluate_transformer(model, data_loader.val_loader, criterion, pad_idx)
        else:  # RNN models
            train_loss = train_rnn(model, data_loader.train_loader, optimizer, criterion, args.clip)
            val_loss, val_accuracy = evaluate_rnn(model, data_loader.val_loader, criterion, pad_idx)
        
        # Track metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Print progress
        print(f"Epoch: {epoch+1}/{args.n_epochs}")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\tVal Loss: {val_loss:.4f}")
        print(f"\tVal Accuracy: {val_accuracy:.4f}")
        
        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            
            # Save model
            model_path = os.path.join(args.save_dir, f'best_{model_type}{config_suffix}_model.pt')
            
            if model_type == 'transformer':
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'input_dim': len(data_loader.english_vocab),
                    'output_dim': len(data_loader.french_vocab),
                    'embed_dim': args.embed_dim,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'dim_feedforward': args.dim_feedforward,
                    'dropout': args.dropout,
                    'pad_idx': pad_idx,
                }
            else:  # RNN models
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'input_dim': len(data_loader.english_vocab),
                    'output_dim': len(data_loader.french_vocab),
                    'embedding_dim': args.embed_dim,
                    'encoder_hidden_dim': args.encoder_hidden_dim,
                    'decoder_hidden_dim': args.decoder_hidden_dim,
                    'n_layers': args.n_layers_rnn,
                    'dropout': args.dropout,
                    'pad_idx': pad_idx,
                }
            
            torch.save(save_dict, model_path)
            print(f"\tBest model saved at {model_path}! (Validation Accuracy: {val_accuracy:.4f})")
            
            # Print example translations for the best model
            print_example_translations(
                model, 
                model_type,
                data_loader, 
                data_loader.english_vocab, 
                data_loader.french_vocab, 
                device, 
                args.num_examples
            )
    
    # Calculate training time
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training completed in {training_time}")
    
    # Save metrics to CSV
    save_training_metrics_csv(history, args.metrics_dir, model_type, config_suffix)
    
    # Save final plots
    save_final_plot(history, args.plot_dir, model_type, config_suffix)
    save_combined_plot(history, args.plot_dir, model_type, config_suffix)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f'final_{model_type}{config_suffix}_model.pt')
    
    if model_type == 'transformer':
        save_dict = {
            'epoch': args.n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'input_dim': len(data_loader.english_vocab),
            'output_dim': len(data_loader.french_vocab),
            'embed_dim': args.embed_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.dropout,
            'pad_idx': pad_idx,
        }
    else:  # RNN models
        save_dict = {
            'epoch': args.n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'input_dim': len(data_loader.english_vocab),
            'output_dim': len(data_loader.french_vocab),
            'embedding_dim': args.embed_dim,
            'encoder_hidden_dim': args.encoder_hidden_dim,
            'decoder_hidden_dim': args.decoder_hidden_dim,
            'n_layers': args.n_layers_rnn,
            'dropout': args.dropout,
            'pad_idx': pad_idx,
        }
    
    torch.save(save_dict, final_model_path)
    print(f"Final model saved at {final_model_path}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    return history, best_val_accuracy


def compare_all_transformer_configs(args, data_loader):
    """Train and compare all transformer configurations."""
    # Define the configurations to test
    layer_options = [1, 2, 4]
    head_options = [2, 4]
    
    results = {}
    
    for num_layers in layer_options:
        for num_heads in head_options:
            config_name = f"_L{num_layers}_H{num_heads}"
            print(f"\n{'='*80}")
            print(f"Training Transformer with {num_layers} layers and {num_heads} heads")
            print(f"{'='*80}\n")
            
            # Set the parameters
            args.num_layers = num_layers
            args.num_heads = num_heads
            
            # Get vocabulary sizes and special tokens
            english_vocab_size, french_vocab_size = data_loader.get_vocab_sizes()
            special_tokens = data_loader.get_special_tokens()
            pad_idx = special_tokens['french']['pad']
            
            # Initialize model
            model = init_transformer_model(
                input_dim=english_vocab_size,
                output_dim=french_vocab_size,
                embed_dim=args.embed_dim,
                num_heads=num_heads,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                src_pad_idx=special_tokens['english']['pad'],
                trg_pad_idx=pad_idx,
                device=device
            )
            
            print(f"The model has {count_parameters(model):,} trainable parameters")
            
            # Train model
            _, best_accuracy = train_model(model, data_loader, 'transformer', args, config_name)
            
            # Store results
            results[config_name] = best_accuracy
    
    # Save results to file
    os.makedirs(args.metrics_dir, exist_ok=True)
    with open(os.path.join(args.metrics_dir, 'transformer_config_comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print comparison
    print("\n\nTransformer Configuration Comparison:")
    print("=" * 40)
    for config, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"Configuration {config}: Accuracy = {accuracy:.4f}")
    
    # Find the best configuration
    best_config = max(results.items(), key=lambda x: x[1])[0]
    print(f"\nBest configuration: {best_config} with accuracy {results[best_config]:.4f}")


def compare_all_models(args, data_loader):
    """Train and compare RNN, RNN with attention, and best Transformer."""
    results = {}
    
    # 1. Train standard RNN
    print(f"\n{'='*80}")
    print(f"Training standard RNN (without attention)")
    print(f"{'='*80}\n")
    
    # Get vocabulary sizes and special tokens
    english_vocab_size, french_vocab_size = data_loader.get_vocab_sizes()
    special_tokens = data_loader.get_special_tokens()
    pad_idx = special_tokens['french']['pad']
    
    # Initialize standard RNN model (imported from model.py, this is the model without attention)
    rnn_model = init_model(
        input_dim=english_vocab_size,
        output_dim=french_vocab_size,
        embedding_dim=args.embed_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        n_layers=args.n_layers_rnn,
        dropout=args.dropout,
        device=device,
        pad_idx=pad_idx
    )
    
    print(f"The RNN model has {count_parameters(rnn_model):,} trainable parameters")
    
    # Train standard RNN
    _, rnn_best_accuracy = train_model(rnn_model, data_loader, 'rnn', args)
    results['rnn'] = rnn_best_accuracy
    
    # 2. Train best Transformer model (from previous configuration search)
    with open(os.path.join(args.metrics_dir, 'transformer_config_comparison.json'), 'r') as f:
        transformer_results = json.load(f)
    
    best_config = max(transformer_results.items(), key=lambda x: float(x[1]))[0]
    num_layers = int(best_config.split('_L')[1].split('_H')[0])
    num_heads = int(best_config.split('_H')[1])
    
    print(f"\n{'='*80}")
    print(f"Training best Transformer (L={num_layers}, H={num_heads})")
    print(f"{'='*80}\n")
    
    args.num_layers = num_layers
    args.num_heads = num_heads
    
    transformer_model = init_transformer_model(
        input_dim=english_vocab_size,
        output_dim=french_vocab_size,
        embed_dim=args.embed_dim,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        src_pad_idx=special_tokens['english']['pad'],
        trg_pad_idx=pad_idx,
        device=device
    )
    
    print(f"The Transformer model has {count_parameters(transformer_model):,} trainable parameters")
    
    # Train best transformer model
    _, transformer_best_accuracy = train_model(transformer_model, data_loader, 'transformer', args, "_best")
    results['transformer'] = transformer_best_accuracy
    
    # Print comparison
    print("\n\nModel Comparison:")
    print("=" * 40)
    for model_type, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_type.capitalize()}: Accuracy = {accuracy:.4f}")
    
    # Find the best model
    best_model = max(results.items(), key=lambda x: x[1])[0]
    print(f"\nBest model: {best_model.capitalize()} with accuracy {results[best_model]:.4f}")
    
    # Save results to file
    with open(os.path.join(args.metrics_dir, 'model_comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories for saving models and plots
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    # Load and prepare the dataset
    data_loader = DatasetLoader(
        csv_path=args.data_path,
        batch_size=args.batch_size,
        augment_ratio=args.augment_ratio
    )
    
    # Get vocabulary sizes
    english_vocab_size, french_vocab_size = data_loader.get_vocab_sizes()
    special_tokens = data_loader.get_special_tokens()
    pad_idx = special_tokens['french']['pad']
    
    print(f"English vocabulary size: {english_vocab_size}")
    print(f"French vocabulary size: {french_vocab_size}")
    
    if args.compare_all_configs:
        # Compare all transformer configurations (1, 2, 4 layers x 2, 4 heads)
        compare_all_transformer_configs(args, data_loader)
        
        # Compare the best transformer with RNN models
        compare_all_models(args, data_loader)
    
    else:
        # Train a single model based on model_type
        if args.model_type == 'transformer':
            # Initialize transformer model
            model = init_transformer_model(
                input_dim=english_vocab_size,
                output_dim=french_vocab_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_encoder_layers=args.num_layers,
                num_decoder_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                src_pad_idx=special_tokens['english']['pad'],
                trg_pad_idx=pad_idx,
                device=device
            )
            
            print(f"The model has {count_parameters(model):,} trainable parameters")
            
            # Train the transformer model
            train_model(model, data_loader, 'transformer', args)
            
        elif args.model_type == 'rnn':
            # Initialize standard RNN model
            model = init_model(
                input_dim=english_vocab_size,
                output_dim=french_vocab_size,
                embedding_dim=args.embed_dim,
                encoder_hidden_dim=args.encoder_hidden_dim,
                decoder_hidden_dim=args.decoder_hidden_dim,
                n_layers=args.n_layers_rnn,
                dropout=args.dropout,
                device=device,
                pad_idx=pad_idx
            )
            
            print(f"The model has {count_parameters(model):,} trainable parameters")
            
            # Train the RNN model
            train_model(model, data_loader, 'rnn', args)