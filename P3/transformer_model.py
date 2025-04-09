import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    Adds information about the position of tokens in the sequence.
    """
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for the English to French translation model.
    """
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, 
                 dim_feedforward=2048, dropout=0.1, max_seq_len=5000):
        super(TransformerEncoder, self).__init__()
        
        # Embedding layer for source sequence
        self.embedding = nn.Embedding(input_dim, embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Scale embeddings
        self.scale = math.sqrt(embed_dim)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Source sequence [batch_size, src_len]
            src_mask: Mask for self-attention (optional)
            src_key_padding_mask: Padding mask for source sequence (optional)
            
        Returns:
            Output of transformer encoder
        """
        # Create embedding and apply positional encoding
        src_embedded = self.embedding(src) * self.scale
        src_embedded = self.positional_encoding(src_embedded)
        
        # Apply transformer encoder
        output = self.transformer_encoder(
            src_embedded, 
            mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        return output


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder for the English to French translation model.
    """
    def __init__(self, output_dim, embed_dim, num_heads, num_layers, 
                 dim_feedforward=2048, dropout=0.1, max_seq_len=5000):
        super(TransformerDecoder, self).__init__()
        
        # Embedding layer for target sequence
        self.embedding = nn.Embedding(output_dim, embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, output_dim)
        
        # Scale embeddings
        self.scale = math.sqrt(embed_dim)
        
    def forward(self, trg, memory, trg_mask=None, memory_mask=None, 
                trg_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            trg: Target sequence [batch_size, trg_len]
            memory: Output from encoder [batch_size, src_len, embed_dim]
            trg_mask: Mask for target sequence (optional)
            memory_mask: Mask for encoder output (optional)
            trg_key_padding_mask: Padding mask for target sequence (optional)
            memory_key_padding_mask: Padding mask for encoder output (optional)
            
        Returns:
            Decoder output and attention weights
        """
        # Create embedding and apply positional encoding
        trg_embedded = self.embedding(trg) * self.scale
        trg_embedded = self.positional_encoding(trg_embedded)
        
        # Apply transformer decoder
        output = self.transformer_decoder(
            trg_embedded, 
            memory, 
            tgt_mask=trg_mask, 
            memory_mask=memory_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project to output dimension
        output = self.output_projection(output)
        
        return output


class TransformerSeq2Seq(nn.Module):
    """
    Full Transformer Sequence-to-Sequence model for English to French translation
    """
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super(TransformerSeq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        """
        Create a mask for padding tokens in the source sequence.
        
        Args:
            src: Source sequence [batch_size, src_len]
            
        Returns:
            Source mask [batch_size, src_len]
        """
        # Create mask for padding tokens (1 for padding, 0 for non-padding)
        src_mask = (src == self.src_pad_idx)
        return src_mask
    
    def make_trg_mask(self, trg):
        """
        Create masks for the target sequence:
        1. Padding mask
        2. Causal mask to prevent attention to future tokens
        
        Args:
            trg: Target sequence [batch_size, trg_len]
            
        Returns:
            Target mask [batch_size, trg_len]
            Target attention mask [trg_len, trg_len]
        """
        # Create mask for padding tokens (1 for padding, 0 for non-padding)
        trg_pad_mask = (trg == self.trg_pad_idx)
        
        # Create causal mask to prevent attention to future tokens
        trg_len = trg.shape[1]
        trg_sub_mask = torch.triu(torch.ones((trg_len, trg_len), device=self.device) * float('-inf'), diagonal=1)
        
        return trg_pad_mask, trg_sub_mask
    
    def forward(self, src, trg):
        """
        Forward pass through the full transformer model.
        
        Args:
            src: Source sequence [batch_size, src_len]
            trg: Target sequence [batch_size, trg_len]
            
        Returns:
            Output tensor of shape [batch_size, trg_len, output_dim]
        """
        # Create masks
        src_padding_mask = self.make_src_mask(src)
        trg_padding_mask, trg_attention_mask = self.make_trg_mask(trg)
        
        # Pass through encoder
        enc_output = self.encoder(src, src_key_padding_mask=src_padding_mask)
        
        # Pass through decoder
        output = self.decoder(
            trg, 
            enc_output, 
            trg_mask=trg_attention_mask,
            trg_key_padding_mask=trg_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        return output


def init_transformer_model(input_dim, output_dim, embed_dim=512, num_heads=8, 
                           num_encoder_layers=6, num_decoder_layers=6, 
                           dim_feedforward=2048, dropout=0.1, 
                           src_pad_idx=0, trg_pad_idx=0, device='cpu'):
    """
    Initialize the Transformer Seq2Seq model.
    
    Args:
        input_dim: Size of source vocabulary
        output_dim: Size of target vocabulary
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Dimension of feedforward network in transformer
        dropout: Dropout rate
        src_pad_idx: Padding index for source vocabulary
        trg_pad_idx: Padding index for target vocabulary
        device: Device to run the model on
        
    Returns:
        Initialized transformer model
    """
    # Initialize encoder and decoder
    encoder = TransformerEncoder(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    decoder = TransformerDecoder(
        output_dim=output_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    # Initialize full model
    model = TransformerSeq2Seq(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        device=device
    ).to(device)
    
    # Initialize model parameters
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
    
    model.apply(initialize_weights)
    
    return model


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def translate_sentence_transformer(model, sentence, english_vocab, french_vocab, 
                                  device, max_length=100):
    """
    Translate a sentence from English to French using the transformer model.
    
    Args:
        model: Trained transformer model
        sentence: English sentence to translate
        english_vocab: English vocabulary object
        french_vocab: French vocabulary object
        device: Device to run inference on
        max_length: Maximum length of the translated sentence
        
    Returns:
        Translated French sentence
    """
    model.eval()
    
    # Ensure proper structure of vocabs
    if isinstance(english_vocab, dict) and 'word2idx' in english_vocab:
        eng_word2idx = english_vocab['word2idx']
    elif hasattr(english_vocab, 'word2idx'):
        eng_word2idx = english_vocab.word2idx
    else:
        eng_word2idx = english_vocab
    
    if isinstance(french_vocab, dict) and 'word2idx' in french_vocab:
        fr_word2idx = french_vocab['word2idx']
        fr_idx2word = french_vocab['idx2word']
    elif hasattr(french_vocab, 'word2idx'):
        fr_word2idx = french_vocab.word2idx
        fr_idx2word = french_vocab.idx2word
    else:
        fr_word2idx = french_vocab
        # Create reverse mapping
        fr_idx2word = {idx: word for word, idx in fr_word2idx.items()}
    
    # Tokenize the sentence
    tokens = [eng_word2idx.get(word, eng_word2idx['<UNK>']) for word in sentence.lower().split()]
    tokens.append(eng_word2idx['<EOS>'])
    
    # Convert to tensor
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    # Create source mask
    src_mask = model.make_src_mask(src_tensor)
    
    # Encode the sentence
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_key_padding_mask=src_mask)
    
    # Start with SOS token
    trg_indexes = [fr_word2idx['<SOS>']]
    
    for i in range(max_length):
        # Convert current target sequence to tensor
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        
        # Create masks
        trg_pad_mask, trg_attn_mask = model.make_trg_mask(trg_tensor)
        
        # Decode
        with torch.no_grad():
            output = model.decoder(
                trg_tensor,
                enc_output,
                trg_mask=trg_attn_mask,
                trg_key_padding_mask=trg_pad_mask,
                memory_key_padding_mask=src_mask
            )
        
        # Get prediction for next token (last position in sequence)
        pred_token = output[:, -1, :].argmax(1).item()
        
        # Add predicted token to sequence
        trg_indexes.append(pred_token)
        
        # Stop if EOS token
        if pred_token == fr_word2idx['<EOS>']:
            break
    
    # Convert indices to tokens and join into a sentence
    trg_tokens = [fr_idx2word[i] for i in trg_indexes]
    
    # Remove special tokens
    trg_tokens = [token for token in trg_tokens if token not in ['<SOS>', '<EOS>']]
    
    # Join tokens into a sentence
    translated_sentence = ' '.join(trg_tokens)
    
    return translated_sentence