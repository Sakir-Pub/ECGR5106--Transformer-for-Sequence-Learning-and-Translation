import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

def load_metrics(transformer_dir, rnn_dir):
    """Load metrics for both transformer and RNN models."""
    # Load transformer metrics
    transformer_files = glob.glob(f"{transformer_dir}/*_metrics.csv")
    transformer_metrics = []
    
    for file in transformer_files:
        df = pd.read_csv(file)
        if not df.empty:
            transformer_metrics.append(df.iloc[0].to_dict())
    
    # Load RNN metrics
    rnn_files = glob.glob(f"{rnn_dir}/*_metrics.csv")
    rnn_metrics = []
    
    for file in rnn_files:
        df = pd.read_csv(file)
        if not df.empty:
            rnn_metrics.append(df.iloc[0].to_dict())
            
    return transformer_metrics, rnn_metrics

def load_histories(transformer_dir, rnn_dir):
    """Load training histories for both transformer and RNN models."""
    # Load transformer histories
    transformer_files = glob.glob(f"{transformer_dir}/*_history.csv")
    transformer_histories = {}
    
    for file in transformer_files:
        base_name = os.path.basename(file).replace('_history.csv', '')
        df = pd.read_csv(file)
        if not df.empty:
            transformer_histories[base_name] = df.to_dict('records')
    
    # Load RNN histories
    rnn_files = glob.glob(f"{rnn_dir}/*_history.csv")
    rnn_histories = {}
    
    for file in rnn_files:
        base_name = os.path.basename(file).replace('_history.csv', '')
        df = pd.read_csv(file)
        if not df.empty:
            rnn_histories[base_name] = df.to_dict('records')
            
    return transformer_histories, rnn_histories

def plot_model_comparison(transformer_metrics, rnn_metrics, metric_key, title, ylabel, output_path, seq_lengths=[20, 30, 50]):
    """Create a bar plot comparing transformer and RNN models on a specific metric."""
    plt.figure(figsize=(15, 8))
    
    # Filter metrics by sequence length and group by model type
    bar_positions = np.arange(len(seq_lengths)) * 3
    bar_width = 0.7
    
    # Transform metrics for easier plotting
    lstm_values = []
    gru_values = []
    transformer_values = []
    
    for seq_len in seq_lengths:
        # Get LSTM value for this sequence length
        lstm_metric = next((m[metric_key] for m in rnn_metrics 
                           if m['Model Type'].upper() == 'LSTM' 
                           and m['Sequence Length'] == seq_len 
                           and m['Hidden Size'] == 128 
                           and m['Num Layers'] == 1), None)
        lstm_values.append(lstm_metric if lstm_metric is not None else 0)
        
        # Get GRU value for this sequence length
        gru_metric = next((m[metric_key] for m in rnn_metrics 
                          if m['Model Type'].upper() == 'GRU' 
                          and m['Sequence Length'] == seq_len 
                          and m['Hidden Size'] == 128 
                          and m['Num Layers'] == 1), None)
        gru_values.append(gru_metric if gru_metric is not None else 0)
        
        # Get Transformer value for this sequence length
        transformer_metric = next((m[metric_key] for m in transformer_metrics 
                                 if m['Sequence Length'] == seq_len 
                                 and m['Hidden Size'] == 128 
                                 and m['Num Layers'] == 2
                                 and m['Num Heads'] == 2), None)
        transformer_values.append(transformer_metric if transformer_metric is not None else 0)
    
    # Plot bars
    plt.bar(bar_positions - bar_width, lstm_values, bar_width, label='LSTM')
    plt.bar(bar_positions, gru_values, bar_width, label='GRU')
    plt.bar(bar_positions + bar_width, transformer_values, bar_width, label='Transformer')
    
    # Labels and formatting
    plt.xlabel('Sequence Length')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(bar_positions, seq_lengths)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def plot_transformer_architecture_comparison(transformer_metrics, metric_key, title, ylabel, output_path):
    """Plot comparison of different transformer architectures (layers and heads)."""
    plt.figure(figsize=(15, 10))
    
    # We'll use sequence length 30 for this comparison
    seq_len = 30
    hidden_size = 128
    
    # Filter relevant metrics
    filtered_metrics = [m for m in transformer_metrics 
                      if m['Sequence Length'] == seq_len 
                      and m['Hidden Size'] == hidden_size]
    
    # Group by layers and heads
    layer_options = sorted(list(set(m['Num Layers'] for m in filtered_metrics)))
    head_options = sorted(list(set(m['Num Heads'] for m in filtered_metrics)))
    
    # Create a 2D grid of values
    values = np.zeros((len(layer_options), len(head_options)))
    
    for i, layers in enumerate(layer_options):
        for j, heads in enumerate(head_options):
            metric = next((m[metric_key] for m in filtered_metrics 
                         if m['Num Layers'] == layers 
                         and m['Num Heads'] == heads), None)
            if metric is not None:
                values[i, j] = metric
    
    # Create bar positions
    x = np.arange(len(layer_options))
    width = 0.8 / len(head_options)
    
    # Plot grouped bars
    for i, heads in enumerate(head_options):
        offset = (i - len(head_options)/2 + 0.5) * width
        plt.bar(x + offset, values[:, i], width, label=f'{heads} heads')
    
    # Labels and formatting
    plt.xlabel('Number of Layers')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, layer_options)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def plot_training_curves(histories, model_names, metric, title, ylabel, output_path):
    """Plot training curves for selected models."""
    plt.figure(figsize=(12, 8))
    
    for name in model_names:
        if name in histories:
            history = histories[name]
            data = [entry[metric] for entry in history]
            epochs = [entry['epoch'] for entry in history]
            plt.plot(epochs, data, label=name)
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def main():
    # Directories
    transformer_data_dir = 'model_data_transformer'
    rnn_data_dir = 'model_data_shakespear'
    output_dir = 'comparison_plots'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics and histories
    print("Loading model metrics and training histories...")
    transformer_metrics, rnn_metrics = load_metrics(transformer_data_dir, rnn_data_dir)
    transformer_histories, rnn_histories = load_histories(transformer_data_dir, rnn_data_dir)
    
    # Combine histories for easier access
    all_histories = {**transformer_histories, **rnn_histories}
    
    print(f"Loaded {len(transformer_metrics)} transformer models and {len(rnn_metrics)} RNN models")
    
    # 1. Compare accuracy across model types and sequence lengths
    plot_model_comparison(
        transformer_metrics, 
        rnn_metrics, 
        'Final Validation Accuracy',
        'Validation Accuracy Comparison',
        'Accuracy',
        f"{output_dir}/accuracy_comparison.png"
    )
    
    # 2. Compare training time across model types and sequence lengths
    plot_model_comparison(
        transformer_metrics, 
        rnn_metrics, 
        'Training Time (s)',
        'Training Time Comparison',
        'Time (seconds)',
        f"{output_dir}/training_time_comparison.png"
    )
    
    # 3. Compare model size across model types and sequence lengths
    plot_model_comparison(
        transformer_metrics, 
        rnn_metrics, 
        'Model Size (MB)',
        'Model Size Comparison',
        'Size (MB)',
        f"{output_dir}/model_size_comparison.png"
    )
    
    # 4. Compare number of parameters across model types and sequence lengths
    plot_model_comparison(
        transformer_metrics, 
        rnn_metrics, 
        'Parameters',
        'Number of Parameters Comparison',
        'Parameters',
        f"{output_dir}/parameters_comparison.png"
    )
    
    # 5. Analyze transformer architectures (varying layers and heads)
    plot_transformer_architecture_comparison(
        transformer_metrics,
        'Final Validation Accuracy',
        'Effect of Transformer Architecture on Accuracy',
        'Accuracy',
        f"{output_dir}/transformer_architecture_accuracy.png"
    )
    
    plot_transformer_architecture_comparison(
        transformer_metrics,
        'Training Time (s)',
        'Effect of Transformer Architecture on Training Time',
        'Time (seconds)',
        f"{output_dir}/transformer_architecture_time.png"
    )
    
    plot_transformer_architecture_comparison(
        transformer_metrics,
        'Parameters',
        'Effect of Transformer Architecture on Model Size',
        'Number of Parameters',
        f"{output_dir}/transformer_architecture_size.png"
    )
    
    # 6. Plot training curves for selected models
    # Select representative models for comparison
    selected_models = [
        'lstm_seq30_h128_l1',
        'gru_seq30_h128_l1',
        'transformer_seq30_h128_l2_heads2'
    ]
    
    # Training loss curves
    plot_training_curves(
        all_histories,
        selected_models,
        'loss',
        'Training Loss Comparison',
        'Loss',
        f"{output_dir}/training_loss_curves.png"
    )
    
    # Validation loss curves
    plot_training_curves(
        all_histories,
        selected_models,
        'val_loss',
        'Validation Loss Comparison',
        'Validation Loss',
        f"{output_dir}/validation_loss_curves.png"
    )
    
    # Validation accuracy curves
    plot_training_curves(
        all_histories,
        selected_models,
        'val_accuracy',
        'Validation Accuracy Comparison',
        'Accuracy',
        f"{output_dir}/validation_accuracy_curves.png"
    )
    
    print(f"All comparison plots saved to {output_dir}/")

if __name__ == "__main__":
    main()