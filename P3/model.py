import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embedding_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, src):
        # src shape: [batch_size, src_len]
        
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [batch_size, src_len, embedding_size]
        
        # outputs shape: [batch_size, src_len, hidden_size * 2]
        # hidden shape: [num_layers * 2, batch_size, hidden_size]
        outputs, hidden = self.gru(embedded)
        
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, encoder_hidden_size, decoder_hidden_size, 
                 num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.output_size = output_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
        # GRU now only takes the embedding as input, not the context vector
        self.gru = nn.GRU(
            embedding_size, 
            decoder_hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer now only uses decoder hidden state and embedding
        self.fc_out = nn.Linear(decoder_hidden_size + embedding_size, output_size)
        
    def forward(self, input, hidden):
        # Make sure hidden is contiguous
        if not hidden.is_contiguous():
            hidden = hidden.contiguous()
        # input shape: [batch_size, 1]
        # hidden shape: [num_layers, batch_size, decoder_hidden_size]
        
        input = input.unsqueeze(1)  # Make sure input has batch_size dimension
        
        # Get embedding of input
        embedded = self.dropout(self.embedding(input))
        # embedded shape: [batch_size, 1, embedding_size]
        
        # Get outputs from GRU
        output, hidden = self.gru(embedded, hidden)
        # output shape: [batch_size, 1, decoder_hidden_size]
        # hidden shape: [num_layers, batch_size, decoder_hidden_size]
        
        # Prepare for prediction
        output = output.squeeze(1)  # [batch_size, decoder_hidden_size]
        embedded = embedded.squeeze(1)  # [batch_size, embedding_size]
        
        # Concatenate output and embedding for prediction
        prediction = self.fc_out(torch.cat((output, embedded), dim=1))
        # prediction shape: [batch_size, output_size]
        
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        
        # Initialize encoder parameter weights
        self.init_weights(self.encoder)
        # Initialize decoder parameter weights
        self.init_weights(self.decoder)
        
    def init_weights(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src shape: [batch_size, src_len]
        # trg shape: [batch_size, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Get encoder outputs and hidden state
        _, hidden = self.encoder(src)
        
        # First input to the decoder is the <SOS> token
        input = trg[:, 0]
        
        # Reshape encoder hidden state for decoder
        hidden = self._init_decoder_hidden(hidden)
        
        for t in range(1, trg_len):
            # Use previous hidden state to get next output
            output, hidden = self.decoder(input, hidden)
            
            # Store output
            outputs[:, t, :] = output
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Use teacher forcing or predicted token
            input = trg[:, t] if teacher_force else top1
        
        return outputs
    
    def _init_decoder_hidden(self, encoder_hidden):
        # Reshape encoder hidden state for decoder
        # encoder_hidden: [num_layers * 2, batch_size, hidden_size]
        
        if self.encoder.num_layers != self.decoder.num_layers:
            # If encoder and decoder have different number of layers
            return torch.zeros(self.decoder.num_layers, encoder_hidden.shape[1], 
                              self.decoder.decoder_hidden_size).to(self.device)
        else:
            # Combine forward and backward hidden states
            num_layers = self.encoder.num_layers
            batch_size = encoder_hidden.shape[1]
            hidden_size = encoder_hidden.shape[2]
            
            # Reshape to get forward and backward states separately
            hidden = encoder_hidden.view(2, num_layers, batch_size, hidden_size)
            
            # Concatenate forward and backward states for each layer
            decoder_hidden = torch.cat((hidden[0], hidden[1]), dim=2)
            
            # Project the concatenated hidden state to decoder hidden size if necessary
            if hidden_size * 2 != self.decoder.decoder_hidden_size:
                # Simple projection by taking the first decoder_hidden_size dimensions
                decoder_hidden = decoder_hidden[:, :, :self.decoder.decoder_hidden_size]
            
            return decoder_hidden


def init_model(input_dim, output_dim, embedding_dim=256, encoder_hidden_dim=256, 
               decoder_hidden_dim=512, n_layers=2, dropout=0.2, device='cpu', pad_idx=0):
    """Initialize the Seq2Seq model with encoder and decoder."""
    
    # Initialize encoder and decoder
    encoder = Encoder(input_dim, embedding_dim, encoder_hidden_dim, n_layers, dropout).to(device)
    decoder = Decoder(output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, 
                     n_layers, dropout).to(device)
    
    # Initialize Seq2Seq model
    model = Seq2Seq(encoder, decoder, device, pad_idx).to(device)
    
    return model


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def translate_sentence(model, sentence, english_vocab, french_vocab, device, max_length=100):
    """Translate a sentence from English to French."""
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
    
    # Encode the sentence
    with torch.no_grad():
        _, hidden = model.encoder(src_tensor)
    
    # Initialize for decoding
    trg_indexes = [fr_word2idx['<SOS>']]
    hidden = model._init_decoder_hidden(hidden)
    
    # Start translating
    for _ in range(max_length):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden)
        
        # Get the predicted token
        pred_token = output.argmax(1).item()
        
        # Add token to outputs
        trg_indexes.append(pred_token)
        
        # If <EOS> token, stop decoding
        if pred_token == fr_word2idx['<EOS>']:
            break
    
    # Convert indices to tokens and join into a sentence
    trg_tokens = [fr_idx2word[i] for i in trg_indexes]
    
    # Remove special tokens <SOS> and <EOS>
    trg_tokens = [token for token in trg_tokens if token not in ['<SOS>', '<EOS>']]
    
    # Join tokens into a sentence
    translated_sentence = ' '.join(trg_tokens)
    
    return translated_sentence