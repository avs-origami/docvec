import math
import os
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset

from paragraphvec.utils import DATA_DIR


class CustomDataset(Dataset):

    def __init__(self, file_path=""):
    
        (self.TotalWords, self.LinesTokens) = self.load_file(file_path)
        self.WordFreq = self.count_words()
        (self.VocabSize, self.Vocabulary) = self.build_vocab()

    
    def load_file(self, file_path:str):
        """Parse lines from a text file into a tokens list."""

        with open(file_path, 'r', encoding='utf-8') as fd:
            next(fd)  # Skip header
            toks = [Tokenize(line) for line in fd]
        
        return (sum(len(tk) for tk in toks), toks)
    

    def count_words(self):
        """Count word frequencies in the dataset."""

        wf = Counter()
        for sent_toks in self.LinesTokens:
            wf.update(sent_toks)
        
        return wf
    

    def build_vocab(self):
        """Build a vocabulary from the word frequencies."""        

        vocab = {word: idx for (idx, word) in enumerate(self.WordFreq.keys())}
        return (len(vocab), vocab)
        
    
    def __len__(self):
        return len(self.LinesTokens)

    
    def __getitem__(self, idx):
    
        tokens = self.LinesTokens[idx]
        token_ids = [self.Vocabulary[token] for token in tokens if token in self.Vocabulary]
    
        return token_ids


def LoadDataset(file_name:str):
    """Loads contents from a file in the *data* directory into a
    torchtext.data.TabularDataset instance.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    dataset = CustomDataset(file_path)

    return dataset


def Tokenize(line="") -> list[str]:
    
    # lower case
    txt = line.strip().lower()

    # keep only alphanumeric and punctations
    txt = re.sub(r'[^A-Za-z0-9(),.!?\'`]', ' ', line)
    
    # remove multiple whitespace characters
    txt = re.sub(r'\s{2,}', ' ', txt)
    
    # punctations to tokens
    txt = re.sub(r'\(', ' ( ', txt)
    txt = re.sub(r'\)', ' ) ', txt)
    txt = re.sub(r',', ' , ', txt)
    # txt = re.sub(r'\.', ' . ', txt)
    # txt = re.sub(r'!', ' ! ', txt)
    txt = re.sub(r'\?', ' ? ', txt)
    
    # split contractions into multiple tokens
    txt = re.sub(r'\'s', ' \'s', txt)
    txt = re.sub(r'\'ve', ' \'ve', txt)
    txt = re.sub(r'n\'t', ' n\'t', txt)
    txt = re.sub(r'\'re', ' \'re', txt)
    txt = re.sub(r'\'d', ' \'d', txt)
    txt = re.sub(r'\'ll', ' \'ll', txt)
    
    return txt.split()


class NCEBatch():
    """Tiny wrapper class to handle batches."""

    def __init__(self, data: dict[torch.Tensor]):
        self.doc_ids = data['doc']
        self.target_noise_ids = data['tgt']
    
    def to(self, device):  
        self.doc_ids = self.doc_ids.to(device)
        self.target_noise_ids = self.target_noise_ids.to(device)

    def pin_memory(self):
        self.doc_ids.pin_memory()
        self.target_noise_ids.pin_memory()
        return self


class DBOWIterableDataset(IterableDataset):
    """Dataset for DBOW (Distributed Bag of Words) training with negative sampling.
    
    Parameters:
    -----------
    dataset: CustomDataset
        The source dataset containing tokenized text
    num_noise_words: int
        Number of noise words to sample for each target word
    """
    def __init__(
        self, 
        dataset: CustomDataset,
        num_noise_words: int
    ):
        super().__init__()
        self.dataset = dataset
        self.num_noise_words = num_noise_words
        
        # Initialize noise distribution (unigram^0.75)
        self.noise_distribution = self._init_noise_distribution()
        
        # Pre-calculate total number of words
        self.total_words = sum(len(tokens) for tokens in dataset.LinesTokens)

    def _init_noise_distribution(self) -> np.ndarray:
        """Initialize the noise distribution using unigram frequencies raised to 0.75 power."""
        freqs = np.zeros(self.dataset.VocabSize)

        for word, freq in self.dataset.WordFreq.items():
            idx = self.dataset.Vocabulary[word]
            freqs[idx] = freq
        
        probs = np.power(freqs, 0.75)
        probs /= np.sum(probs)

        return probs

    def _sample_noise(self) -> torch.Tensor:
        """Sample noise words from the noise distribution."""
        noise_ids = np.random.choice(
            self.dataset.VocabSize,
            size=self.num_noise_words,
            p=self.noise_distribution
        )

        return torch.tensor(noise_ids, dtype=torch.long)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Calculate per-worker document range
        if worker_info is None:  # Single worker
            doc_range = range(len(self.dataset))
        else:  # Multiple workers
            per_worker = int(np.ceil(len(self.dataset) / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.dataset))
            doc_range = range(start, end)

        # Iterate through documents
        for doc_id in doc_range:
            tokens = self.dataset.LinesTokens[doc_id]
            
            # Skip empty documents
            if not tokens:
                continue
                
            # For each word in document
            for token in tokens:
                if token not in self.dataset.Vocabulary:
                    continue
                    
                target_id = self.dataset.Vocabulary[token]
                
                # Sample noise words
                noise_ids = self._sample_noise()
                
                # Combine target and noise ids
                target_noise = torch.cat([
                    torch.tensor([target_id], dtype=torch.long),
                    noise_ids
                ])
                
                yield {
                    'doc': torch.tensor(doc_id, dtype=torch.long),
                    'tgt': target_noise,
                }

    def __len__(self) -> int:
        """Return the number of batches that will be generated.
        Note: For IterableDataset, this returns the total number of samples,
        which DataLoader will divide by batch_size to get number of batches."""
        if not hasattr(self, '_length'):
            # Count only valid tokens (those in vocabulary)
            self._length = sum(
                sum(1 for token in doc if token in self.dataset.Vocabulary)
                for doc in self.dataset.LinesTokens
            )
        return self._length


def create_dbow_dataloader(
    dataset: CustomDataset,
    batch_size: int,
    num_noise_words: int,
    num_workers: int = 0,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for DBOW training.
    
    Parameters:
    -----------
    dataset: CustomDataset
        The source dataset
    batch_size: int
        Number of samples per batch
    num_noise_words: int
        Number of noise words to sample
    num_workers: int
        Number of worker processes for data loading
    pin_memory: bool
        Whether to pin memory for faster GPU transfer
        
    Returns:
    --------
    torch.utils.data.DataLoader
        DataLoader that yields batches of DBOW training data
    """
    dbow_dataset = DBOWIterableDataset(
        dataset=dataset,
        num_noise_words=num_noise_words
    )
    
    return torch.utils.data.DataLoader(
        dataset=dbow_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )