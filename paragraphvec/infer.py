import torch
import torch.nn as nn

class DM(nn.Module):
    # Existing code...

    def infer_vector(self, tokens, vocabulary, steps=10, lr=0.025):
        """
        Infer a paragraph vector for an unseen document.

        Parameters:
        tokens (list of str): Tokenized words of the document.
        vocabulary (dict): Mapping from words to indices.
        steps (int): Number of optimization steps.
        lr (float): Learning rate for optimization.

        Returns:
        torch.Tensor: Inferred paragraph vector.
        """
        self.eval()  # Set the model to evaluation mode

        # Initialize a new paragraph vector
        doc_vector = nn.Parameter(torch.randn(self._D.size(1)), requires_grad=True)
        optimizer = torch.optim.SGD([doc_vector], lr=lr)

        # Convert tokens to indices
        context_ids = [vocabulary[token] for token in tokens if token in vocabulary]
        context_ids = torch.LongTensor([context_ids])

        # Move tensors to the same device as model parameters
        if next(self.parameters()).is_cuda:
            context_ids = context_ids.cuda()
            doc_vector = doc_vector.cuda()

        # Inference loop
        for _ in range(steps):
            optimizer.zero_grad()
            # Compute the combined vector
            x = doc_vector + torch.sum(self._W[context_ids], dim=1)
            # Compute scores
            scores = torch.matmul(x, self._O)
            # Negative sampling is not applicable here, use softmax
            log_probs = nn.functional.log_softmax(scores, dim=1)
            # We don't have a target, so minimize negative log likelihood
            loss = -log_probs.mean()
            loss.backward()
            optimizer.step()

        return doc_vector.detach()
    

from data import Tokenize
from models import DM

# Load the trained model
model = DM(vec_dim, num_docs, num_words)
model.load_state_dict(torch.load('path_to_trained_model.pt'))

# Use the same vocabulary from your dataset
vocabulary = dataset.Vocabulary

# Input document text
document_text = "Your new document text goes here."
tokens = Tokenize(document_text)

# Infer the vector
document_vector = model.infer_vector(tokens, vocabulary)

# The resulting `document_vector` is a torch.Tensor containing the embedding











import torch
from torch.utils.data import IterableDataset
import csv

class CSVDataset(IterableDataset):
    """
    An IterableDataset that reads data from a CSV file line by line.
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self._iterator()
        else:
            return self._worker_iterator(worker_info.id, worker_info.num_workers)

    def _iterator(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                yield self._process_row(row)

    def _worker_iterator(self, worker_id, num_workers):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx % num_workers == worker_id:
                    yield self._process_row(row)

    def _process_row(self, row):
        # Implement any preprocessing here
        return row


from torch.utils.data import DataLoader

dataset = CSVDataset('path/to/your/large_file.csv')
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

for batch in dataloader:
    # Training loop
    pass