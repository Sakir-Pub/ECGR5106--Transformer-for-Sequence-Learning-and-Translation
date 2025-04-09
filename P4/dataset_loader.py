import numpy as np
import pandas as pd
import string
import re
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from copy import deepcopy

class TranslationDataset(Dataset):
    def __init__(self, french_sentences, english_sentences):
        self.french_sentences = french_sentences
        self.english_sentences = english_sentences
    
    def __len__(self):
        return len(self.french_sentences)
    
    def __getitem__(self, idx):
        return {
            "french": self.french_sentences[idx],
            "english": self.english_sentences[idx]
        }

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "< SOS >": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "< SOS >", 2: "<EOS>", 3: "<UNK>"}
        self.word_count = {}
        self.idx = 4
    
    def build_vocabulary(self, sentences, min_freq=1):
        # Count word frequencies
        word_counts = Counter()
        for sentence in sentences:
            for word in sentence.split():
                word_counts[word] += 1
        
        # Add words to vocabulary if they meet minimum frequency
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.word_count[word] = count
                self.idx += 1
    
    def sentence_to_indices(self, sentence):
        indices = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence.split()]
        # Add EOS token
        indices.append(self.word2idx["<EOS>"])
        return indices
    
    def indices_to_sentence(self, indices):
        words = [self.idx2word.get(idx, "<UNK>") for idx in indices]
        # Remove special tokens
        words = [word for word in words if word not in ["<PAD>", "< SOS >", "<EOS>"]]
        return " ".join(words)
    
    def __len__(self):
        return len(self.word2idx)

class DataAugmenter:
    def __init__(self, french_sentences, english_sentences, seed=42):
        """
        Initialize data augmenter for French-English translation pairs
        
        Args:
            french_sentences: List of preprocessed French sentences
            english_sentences: List of preprocessed English sentences
            seed: Random seed for reproducibility
        """
        self.french_sentences = french_sentences
        self.english_sentences = english_sentences
        random.seed(seed)
    
    def swap_words_within_sentence(self, sentences, swap_prob=0.2, max_swaps=2):
        """
        Swap random words within each sentence
        
        Args:
            sentences: List of preprocessed sentences
            swap_prob: Probability of performing swaps for each sentence
            max_swaps: Maximum number of swaps per sentence
            
        Returns:
            List of augmented sentences
        """
        augmented_sentences = []
        
        for sentence in sentences:
            if random.random() > swap_prob:
                augmented_sentences.append(sentence)
                continue
                
            words = sentence.split()
            if len(words) < 3:  # Need at least 3 words to swap
                augmented_sentences.append(sentence)
                continue
            
            # Determine number of swaps for this sentence (1 to max_swaps)
            num_swaps = random.randint(1, min(max_swaps, len(words) // 2))
            
            for _ in range(num_swaps):
                # Choose two different positions to swap
                pos1, pos2 = random.sample(range(len(words)), 2)
                # Swap the words
                words[pos1], words[pos2] = words[pos2], words[pos1]
            
            augmented_sentences.append(" ".join(words))
        
        return augmented_sentences
    
    def swap_words_between_sentences(self, sentences, swap_prob=0.15, max_swaps=2):
        """
        Swap random words between different sentences
        
        Args:
            sentences: List of preprocessed sentences
            swap_prob: Probability of performing swaps for each sentence
            max_swaps: Maximum number of swaps per sentence
            
        Returns:
            List of augmented sentences
        """
        augmented_sentences = [s for s in sentences]  # Create a copy
        
        # Only augment if there are enough sentences
        if len(sentences) < 2:
            return augmented_sentences
        
        for i in range(len(sentences)):
            if random.random() > swap_prob:
                continue
                
            words_i = augmented_sentences[i].split()
            if len(words_i) < 3:  # Need at least 3 words in the sentence
                continue
                
            # Determine number of swaps for this sentence
            num_swaps = random.randint(1, min(max_swaps, len(words_i) // 3))
            
            for _ in range(num_swaps):
                # Choose a different sentence
                j = random.choice([k for k in range(len(sentences)) if k != i])
                words_j = augmented_sentences[j].split()
                
                if len(words_j) < 3:
                    continue
                
                # Choose a word position in each sentence
                pos_i = random.randint(0, len(words_i) - 1)
                pos_j = random.randint(0, len(words_j) - 1)
                
                # Swap the words
                words_i[pos_i], words_j[pos_j] = words_j[pos_j], words_i[pos_i]
                
                # Update both sentences
                augmented_sentences[i] = " ".join(words_i)
                augmented_sentences[j] = " ".join(words_j)
                
                # Update words_i for next iteration
                words_i = augmented_sentences[i].split()
        
        return augmented_sentences
    
    def generate_augmented_data(self, augment_ratio=0.5):
        """
        Generate augmented data based on the original dataset
        
        Args:
            augment_ratio: Ratio of augmented data to original data
            
        Returns:
            Tuple of (french_sentences, english_sentences) with augmentations
        """
        num_augmented = int(len(self.french_sentences) * augment_ratio)
        if num_augmented == 0:
            return self.french_sentences, self.english_sentences
        
        # Randomly select indices to augment
        indices_to_augment = random.sample(range(len(self.french_sentences)), num_augmented)
        
        # Create copies of original sentences
        aug_french = [s for s in self.french_sentences]
        aug_english = [s for s in self.english_sentences]
        
        # Get sentences to augment
        to_augment_french = [self.french_sentences[i] for i in indices_to_augment]
        to_augment_english = [self.english_sentences[i] for i in indices_to_augment]
        
        # Apply augmentations (50% within-sentence swaps, 50% between-sentence swaps)
        within_swap_count = num_augmented // 2
        
        # Within-sentence swaps
        within_indices = indices_to_augment[:within_swap_count]
        within_french = self.swap_words_within_sentence(
            [self.french_sentences[i] for i in within_indices]
        )
        within_english = self.swap_words_within_sentence(
            [self.english_sentences[i] for i in within_indices]
        )
        
        # Between-sentence swaps - use all sentences as candidates but update only the selected indices
        between_indices = indices_to_augment[within_swap_count:]
        
        # Create temporary lists for between-sentence swaps
        temp_french = [s for s in self.french_sentences]
        temp_english = [s for s in self.english_sentences]
        
        # Apply between-sentence swaps for French
        between_french = self.swap_words_between_sentences(temp_french)
        between_french = [between_french[i] for i in between_indices]
        
        # Apply between-sentence swaps for English (independently)
        between_english = self.swap_words_between_sentences(temp_english)
        between_english = [between_english[i] for i in between_indices]
        
        # Update augmented sentences
        for idx, new_sent in zip(within_indices, within_french):
            aug_french[idx] = new_sent
        
        for idx, new_sent in zip(within_indices, within_english):
            aug_english[idx] = new_sent
            
        for idx, new_sent in zip(between_indices, between_french):
            aug_french[idx] = new_sent
            
        for idx, new_sent in zip(between_indices, between_english):
            aug_english[idx] = new_sent
        
        print(f"Generated {num_augmented} augmented sentence pairs")
        return aug_french, aug_english

class DatasetLoader:
    def __init__(self, csv_path, batch_size=32, augment_ratio=0.5, random_seed=42):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.augment_ratio = augment_ratio
        self.random_seed = random_seed
        
        self.french_vocab = Vocabulary()
        self.english_vocab = Vocabulary()
        
        self.max_french_length = 0
        self.max_english_length = 0
        
        # Load and preprocess original data
        self.original_data = self._load_data()
        
        # Create augmented data for training
        self.train_data = self._create_augmented_data() if augment_ratio > 0 else self.original_data
        
        # Use original data for validation
        self.val_data = self.original_data
        
        # Create PyTorch DataLoader objects
        self.train_loader = self._create_dataloader(self.train_data, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_data, shuffle=False)
    
    def _load_data(self):
        """Load and preprocess data from CSV file"""
        print(f"Loading data from {self.csv_path}")
        
        # Load CSV file
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} sentence pairs")
        
        # Preprocess data
        self.french_sentences = [self._preprocess_text(text) for text in df['French']]
        self.english_sentences = [self._preprocess_text(text) for text in df['English']]
        
        # Build vocabularies
        print("Building French vocabulary...")
        self.french_vocab.build_vocabulary(self.french_sentences)
        print(f"French vocabulary size: {len(self.french_vocab)}")
        
        print("Building English vocabulary...")
        self.english_vocab.build_vocabulary(self.english_sentences)
        print(f"English vocabulary size: {len(self.english_vocab)}")
        
        # Convert sentences to indices
        french_indices = [self.french_vocab.sentence_to_indices(sentence) 
                          for sentence in self.french_sentences]
        english_indices = [self.english_vocab.sentence_to_indices(sentence) 
                         for sentence in self.english_sentences]
        
        # Update max lengths
        self.max_french_length = max(len(indices) for indices in french_indices)
        self.max_english_length = max(len(indices) for indices in english_indices)
        
        return list(zip(french_indices, english_indices))
    
    def _create_augmented_data(self):
        """Create augmented data for training"""
        print(f"Creating augmented data with ratio {self.augment_ratio}...")
        
        # Initialize data augmenter
        augmenter = DataAugmenter(
            self.french_sentences, 
            self.english_sentences,
            seed=self.random_seed
        )
        
        # Generate augmented sentences
        aug_french, aug_english = augmenter.generate_augmented_data(self.augment_ratio)
        
        # Convert augmented sentences to indices
        aug_french_indices = [self.french_vocab.sentence_to_indices(sentence) 
                              for sentence in aug_french]
        aug_english_indices = [self.english_vocab.sentence_to_indices(sentence) 
                             for sentence in aug_english]
        
        # Update max lengths with augmented data
        self.max_french_length = max(self.max_french_length, 
                                     max(len(indices) for indices in aug_french_indices))
        self.max_english_length = max(self.max_english_length, 
                                    max(len(indices) for indices in aug_english_indices))
        
        return list(zip(aug_french_indices, aug_english_indices))
    
    def _preprocess_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Add spacing between punctuation
        text = re.sub(f"([{string.punctuation}])", r" \1 ", text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _create_dataloader(self, data, shuffle=True):
        """Create a PyTorch DataLoader"""
        french_data = [item[0] for item in data]
        english_data = [item[1] for item in data]
        
        dataset = TranslationDataset(french_data, english_data)
        
        def collate_fn(batch):
            french_sequences = [torch.tensor(item["french"]) for item in batch]
            english_sequences = [torch.tensor(item["english"]) for item in batch]
            
            # Add SOS token for decoder input
            english_input = [torch.tensor([self.english_vocab.word2idx["< SOS >"]] + seq.tolist()[:-1])  # Exclude EOS
                           for seq in english_sequences]
            
            # Pad sequences
            padded_french = pad_sequence(french_sequences, batch_first=True, 
                                          padding_value=self.french_vocab.word2idx["<PAD>"])
            padded_english = pad_sequence(english_sequences, batch_first=True, 
                                         padding_value=self.english_vocab.word2idx["<PAD>"])
            padded_english_input = pad_sequence(english_input, batch_first=True, 
                                              padding_value=self.english_vocab.word2idx["<PAD>"])
            
            return {
                "encoder_inputs": padded_french,
                "decoder_inputs": padded_english_input,
                "decoder_targets": padded_english
            }
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )
    
    def get_vocab_sizes(self):
        """Return vocabulary sizes for model initialization"""
        return len(self.french_vocab), len(self.english_vocab)
    
    def get_max_lengths(self):
        """Return maximum sequence lengths"""
        return self.max_french_length, self.max_english_length
    
    def get_special_tokens(self):
        """Return special token indices for both vocabularies"""
        return {
            "french": {
                "pad": self.french_vocab.word2idx["<PAD>"],
                "sos": self.french_vocab.word2idx["< SOS >"],
                "eos": self.french_vocab.word2idx["<EOS>"],
                "unk": self.french_vocab.word2idx["<UNK>"]
            },
            "english": {
                "pad": self.english_vocab.word2idx["<PAD>"],
                "sos": self.english_vocab.word2idx["< SOS >"],
                "eos": self.english_vocab.word2idx["<EOS>"],
                "unk": self.english_vocab.word2idx["<UNK>"]
            }
        }

# Example usage
if __name__ == "__main__":
    # Path to your CSV file
    csv_path = "/home/anabil/Development/PhD Courses/ECGR 5106/HW4/Dataset/english_french_phrases.csv"
    
    # Initialize the dataset loader with data augmentation (50% more augmented data)
    loader = DatasetLoader(csv_path, batch_size=16, augment_ratio=0.5)
    
    print("\nTraining data (with augmentations):")
    print(f"Number of training samples: {len(loader.train_data)}")
    
    # Get a batch from the training loader
    for batch in loader.train_loader:
        print(f"Encoder inputs shape: {batch['encoder_inputs'].shape}")
        print(f"Decoder inputs shape: {batch['decoder_inputs'].shape}")
        print(f"Decoder targets shape: {batch['decoder_targets'].shape}")
        
        # Display a sample
        sample_idx = 0
        french_indices = batch['encoder_inputs'][sample_idx].tolist()
        english_indices = batch['decoder_targets'][sample_idx].tolist()
        
        french_sentence = loader.french_vocab.indices_to_sentence(french_indices)
        english_sentence = loader.english_vocab.indices_to_sentence(english_indices)
        
        print(f"French: {french_sentence}")
        print(f"English: {english_sentence}")
        break
    
    print("\nValidation data (original data):")
    print(f"Number of validation samples: {len(loader.val_data)}")
    for batch in loader.val_loader:
        print(f"Encoder inputs shape: {batch['encoder_inputs'].shape}")
        print(f"Decoder inputs shape: {batch['decoder_inputs'].shape}")
        print(f"Decoder targets shape: {batch['decoder_targets'].shape}")
        break