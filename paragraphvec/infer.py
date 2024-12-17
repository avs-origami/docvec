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

