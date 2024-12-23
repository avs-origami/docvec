import os
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from paragraphvec.utils import DATA_DIR

def LoadDataset(num_noise:int, file_name:str):
    """Loads contents from a file in the *data* directory into a
    torchtext.data.TabularDataset instance.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    dataset = CustomDataset(num_noise, file_path)

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

class CustomDataset(Dataset):
    def __init__(self, num_noise_words, file_path=""):
        (self.TotalWords, self.LinesTokens) = self.load_file(file_path)
        self.WordFreq = self.count_words()
        (self.VocabSize, self.Vocabulary) = self.build_vocab()

        self.in_doc_pos = 0
        self.num_noise_words = num_noise_words
        self.noise_dist = self._init_noise_distribution()

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

    def get_token_ids(self, idx):
        tokens = self.LinesTokens[idx]
        token_ids = [self.Vocabulary[token] for token in tokens if token in self.Vocabulary]
    
        return token_ids
    
    def _init_noise_distribution(self):
        # we use a unigram distribution raised to the 3/4rd power,
        # as proposed by T. Mikolov et al. in Distributed Representations
        # of Words and Phrases and their Compositionality
        freqs = np.zeros(self.VocabSize)

        for (word, freq) in self.WordFreq.items():
            idx = self.Vocabulary[word]
            freqs[idx] = freq

        probs = np.power(freqs, 0.75)
        probs /= np.sum(probs)
        
        return probs


    def _sample_noise(self) -> list[int]:
        x = np.random.choice(self.VocabSize, self.num_noise_words, p=self.noise_dist)
        return x.tolist()
    
    def __getitem__(self, doc_id):
        """Get a single item in a batch consisting of a target
        id and several noise ids."""
    
        tokens = self.LinesTokens[doc_id]
        current_noise = self._sample_noise()

        if self.in_doc_pos >= len(tokens):
            self.in_doc_pos = np.random.randint(0, len(tokens))

        target_word = tokens[self.in_doc_pos]
        current_noise.insert(0, self.Vocabulary[target_word])

        self.in_doc_pos += 1

        return {
            'doc': torch.LongTensor([doc_id]).cuda(),
            'tgt': torch.LongTensor(current_noise).cuda(),
        }
    
    def __len__(self):
        return len(self.LinesTokens)