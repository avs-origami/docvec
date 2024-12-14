import multiprocessing
import os
import re
import signal
from math import ceil

import numpy as np
import torch
from numpy.random import choice

import numpy as np
from collections import Counter
from torch.utils.data import Dataset

from paragraphvec.utils import DATA_DIR


class CustomDataset(Dataset):

    def __init__(self, file_path:str):
    
        self.counter = Counter()
        self.examples = []
        self.vocab = dict()
        self.vocab_size = 0
    
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
    
            for line in f:
                text = line.strip()
                tokens = _tokenize_str(text)
                
                self.examples.append(tokens)
    
        self.build_vocab()

    
    def build_vocab(self):

        for tokens in self.examples:
            self.counter.update(tokens)
        
        self.vocab = {word: idx for idx, word in enumerate(self.counter.keys())}
        self.vocab_size = len(self.vocab)
        
        #self.stoi = self.vocab  # string to index mapping
        #self.itos = {idx: word for word, idx in self.vocab.items()}  # index to string mapping

    
    def __len__(self):
        return len(self.examples)

    
    def __getitem__(self, idx):
    
        tokens = self.examples[idx]
        token_ids = [self.vocab[token] for token in tokens if token in self.vocab]
    
        return token_ids


def load_dataset(file_name:str):
    """Loads contents from a file in the *data* directory into a
    torchtext.data.TabularDataset instance.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    dataset = CustomDataset(file_path)

    return dataset


def _tokenize_str(str_):
    
    # keep only alphanumeric and punctations
    str_ = re.sub(r'[^A-Za-z0-9(),.!?\'`]', ' ', str_)
    
    # remove multiple whitespace characters
    str_ = re.sub(r'\s{2,}', ' ', str_)
    
    # punctations to tokens
    str_ = re.sub(r'\(', ' ( ', str_)
    str_ = re.sub(r'\)', ' ) ', str_)
    str_ = re.sub(r',', ' , ', str_)
    str_ = re.sub(r'\.', ' . ', str_)
    str_ = re.sub(r'!', ' ! ', str_)
    str_ = re.sub(r'\?', ' ? ', str_)
    
    # split contractions into multiple tokens
    str_ = re.sub(r'\'s', ' \'s', str_)
    str_ = re.sub(r'\'ve', ' \'ve', str_)
    str_ = re.sub(r'n\'t', ' n\'t', str_)
    str_ = re.sub(r'\'re', ' \'re', str_)
    str_ = re.sub(r'\'d', ' \'d', str_)
    str_ = re.sub(r'\'ll', ' \'ll', str_)
    
    # lower case
    return str_.strip().lower().split()


class NCEData(object):
    """An infinite, parallel (multiprocess) batch generator for
    noise-contrastive estimation of word vector models.

    Parameters
    ----------
    dataset: torchtext.data.TabularDataset
        Dataset from which examples are generated. A column labeled *text*
        is expected and should be comprised of a list of tokens. Each row
        should represent a single document.

    batch_size: int
        Number of examples per single gradient update.

    context_size: int
        Half the size of a neighbourhood of target words (i.e. how many
        words left and right are regarded as context).

    num_noise_words: int
        Number of noise words to sample from the noise distribution.

    max_size: int
        Maximum number of pre-generated batches.

    num_workers: int
        Number of jobs to run in parallel. If value is set to -1, total number
        of machine CPUs is used.
    """
    # code inspired by parallel generators in https://github.com/fchollet/keras
    def __init__(self, dataset, batch_size, context_size,
                 num_noise_words, max_size, num_workers):
        self.max_size = max_size

        self.num_workers = num_workers if num_workers != -1 else os.cpu_count()
        if self.num_workers is None:
            self.num_workers = 1

        self._generator = _NCEGenerator(
            dataset,
            batch_size,
            context_size,
            num_noise_words,
            _NCEGeneratorState(context_size))

        self._queue = None
        self._stop_event = None
        self._processes = []

    def __len__(self):
        return len(self._generator)

    def vocabulary_size(self):
        return self._generator.vocabulary_size()

    def start(self):
        """Starts num_worker processes that generate batches of data."""
        self._queue = multiprocessing.Queue(maxsize=self.max_size)
        self._stop_event = multiprocessing.Event()

        for _ in range(self.num_workers):
            process = multiprocessing.Process(target=self._parallel_task)
            process.daemon = True
            self._processes.append(process)
            process.start()

    def _parallel_task(self):
        while not self._stop_event.is_set():
            try:
                batch = self._generator.next()
                # queue blocks a call to put() until a free slot is available
                self._queue.put(batch)
            except KeyboardInterrupt:
                self._stop_event.set()

    def get_generator(self):
        """Returns a generator that yields batches of data."""
        while self._is_running():
            yield self._queue.get()

    def stop(self):
        """Terminates all processes that were created with start()."""
        if self._is_running():
            self._stop_event.set()

        for process in self._processes:
            if process.is_alive():
                os.kill(process.pid, signal.SIGINT)
                process.join()

        if self._queue is not None:
            self._queue.close()

        self._queue = None
        self._stop_event = None
        self._processes = []

    def _is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()


class _NCEGenerator(object):
    """An infinite, process-safe batch generator for noise-contrastive
    estimation of word vector models.

    Parameters
    ----------
    state: paragraphvec.data._NCEGeneratorState
        Initial (indexing) state of the generator.

    For other parameters see the NCEData class.
    """
    def __init__(self, dataset, batch_size, context_size, num_noise_words, state):

        self.dataset = dataset
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_noise_words = num_noise_words

        self._vocabulary = self.dataset.vocab
        self._vocab_size = self.dataset.vocab_size
        self._init_noise_distribution()
        self._state = state

    
    def _init_noise_distribution(self):
        # we use a unigram distribution raised to the 3/4rd power,
        # as proposed by T. Mikolov et al. in Distributed Representations
        # of Words and Phrases and their Compositionality
        freqs = np.zeros(self._vocab_size)

        for word, freq in self.dataset.counter.items():
            idx = self._word_to_index(word)
            freqs[idx] = freq

        probs = np.power(freqs, 0.75)
        probs /= np.sum(probs)
        self._noise_distribution = probs


    def _sample_noise(self):
        
        return choice(
            self._vocab_size, self.num_noise_words, p=self._noise_distribution
            ).tolist()
        

    def __len__(self):
        num_examples = sum(self._num_examples_in_doc(i) for i in range(len(self.dataset)))
        return ceil(num_examples / self.batch_size)


    def vocabulary_size(self):
        return self._vocab_size
    

    def next(self):
        """Updates state for the next process in a process-safe manner
        and generates the current batch."""

        prev_doc_id, prev_in_doc_pos = self._state.update_state(
            self.dataset,
            self.batch_size,
            self.context_size,
            self._num_examples_in_doc)

        # generate the actual batch
        batch = _NCEBatch(self.context_size)

        while len(batch) < self.batch_size:
            if prev_doc_id == len(self.dataset):
                batch.torch_()
                return batch
        
            tokens = self.dataset.examples[prev_doc_id]
        
            if prev_in_doc_pos <= (len(tokens) - 1 - self.context_size):
                self._add_example_to_batch(prev_doc_id, prev_in_doc_pos, batch)
                prev_in_doc_pos += 1
            else:
                prev_doc_id += 1
                prev_in_doc_pos = self.context_size

        batch.torch_()
        return batch


    def _num_examples_in_doc(self, doc_id=0, in_doc_pos=None):
    
        tokens = self.dataset.examples[doc_id]
    
        if in_doc_pos is not None:
            if len(tokens) - in_doc_pos >= self.context_size + 1:
                return len(tokens) - in_doc_pos - self.context_size
    
            return 0

        if len(tokens) >= 2 * self.context_size + 1:
            return len(tokens) - 2 * self.context_size
    
        return 0
    

    def _add_example_to_batch(self, doc_id, in_doc_pos, batch):
        
        tokens = self.dataset.examples[doc_id]
        batch.doc_ids.append(doc_id)

        # Sample from the noise distribution
        current_noise = self._sample_noise()
        target_word = tokens[in_doc_pos]
        current_noise.insert(0, self._word_to_index(target_word))
        batch.target_noise_ids.append(current_noise)

        if self.context_size == 0:
            return

        current_context = []
        context_indices = (
            in_doc_pos + diff for diff in
            range(-self.context_size, self.context_size + 1)
            if diff != 0
        )

        for i in context_indices:
            context_word = tokens[i]
            context_id = self._word_to_index(context_word)
            current_context.append(context_id)
        
        batch.context_ids.append(current_context)


    def _word_to_index(self, word):
        return self._vocabulary[word]
    


class _NCEGeneratorState(object):
    """Batch generator state that is represented with a document id and
    in-document position. It abstracts a process-safe indexing mechanism."""
    
    def __init__(self, context_size):
        # use raw values because both indices have
        # to manually be locked together
        self._doc_id = multiprocessing.RawValue('i', 0)
        self._in_doc_pos = multiprocessing.RawValue('i', context_size)
        self._lock = multiprocessing.Lock()

    
    def update_state(self, dataset, batch_size, context_size, num_examples_in_doc):
        """Returns current indices and computes new indices for the next process."""

        with self._lock:
            doc_id = self._doc_id.value
            in_doc_pos = self._in_doc_pos.value
            self._advance_indices(dataset, batch_size, context_size, num_examples_in_doc)
            
            return doc_id, in_doc_pos


    def _advance_indices(self, dataset, batch_size, context_size, num_examples_in_doc):

        doc_id = self._doc_id.value
        in_doc_pos = self._in_doc_pos.value
        num_examples = num_examples_in_doc(doc_id, in_doc_pos)

        if num_examples > batch_size:
            # more examples in the current document
            self._in_doc_pos.value += batch_size
            return

        if num_examples == batch_size:
            # just enough examples in the current document
            if self._doc_id.value < len(dataset) - 1:
                self._doc_id.value += 1
            else:
                self._doc_id.value = 0
            self._in_doc_pos.value = context_size
            return

        while num_examples < batch_size:
            if self._doc_id.value == len(dataset) - 1:
                # last document: reset indices
                self._doc_id.value = 0
                self._in_doc_pos.value = context_size
                return

            self._doc_id.value += 1
            num_examples += num_examples_in_doc(self._doc_id.value)

        sent_len = len(dataset[self._doc_id.value])
        x = (num_examples - batch_size)        
        self._in_doc_pos.value = sent_len - context_size - x


class _NCEBatch(object):

    def __init__(self, context_size):
    
        self.context_ids = [] if context_size > 0 else None
        self.doc_ids = []
        self.target_noise_ids = []

    
    def __len__(self):
        return len(self.doc_ids)

    
    def torch_(self):
    
        if self.context_ids is not None:
            self.context_ids = torch.LongTensor(self.context_ids)
    
        self.doc_ids = torch.LongTensor(self.doc_ids)
        self.target_noise_ids = torch.LongTensor(self.target_noise_ids)

    
    def cuda_(self):
    
        if self.context_ids is not None:
            self.context_ids = self.context_ids.cuda()
    
        self.doc_ids = self.doc_ids.cuda()
        self.target_noise_ids = self.target_noise_ids.cuda()


    def to(self, device):

        if device.type == 'cuda':
            self.cuda_()
        else:
            self.torch_()