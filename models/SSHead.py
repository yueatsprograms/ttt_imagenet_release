from torch import nn
import torch.nn.functional as F
import math

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head

	def forward(self, x):
		return self.head(self.ext(x))

def extractor_from_layer4(net):
	layers = [net.conv1, net.bn1, net.relu, net.maxpool,
				 net.layer1, net.layer2, net.layer3, net.layer4, 
					net.avgpool, ViewFlatten()]
	return nn.Sequential(*layers)

def extractor_from_layer3(net):
	layers = [net.conv1, net.bn1, net.relu, net.maxpool,
				 net.layer1, net.layer2, net.layer3]
	return nn.Sequential(*layers)

def extractor_from_layer2(net):
	layers = [net.conv1, net.bn1, net.relu, net.maxpool,
				 net.layer1, net.layer2]
	return nn.Sequential(*layers)
