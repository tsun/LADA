import torch
import torch.nn.functional as F

class ConditionalEntropyLoss(torch.nn.Module):
	"""
	Conditional entropy loss utility class
	"""
	def __init__(self):
		super(ConditionalEntropyLoss, self).__init__()

	def forward(self, x):
		b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
		b = b.sum(dim=1)
		return -1.0 * b.mean(dim=0)
