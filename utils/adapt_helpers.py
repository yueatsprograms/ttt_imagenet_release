import torch
import torch.nn as nn

from utils.train_helpers import *
from utils.rotation import rotate_batch, rotate_single_with_label

def trerr_single(model, image):
	model.eval()
	labels = torch.LongTensor([0, 1, 2, 3])
	inputs = []
	for label in labels:
		inputs.append(rotate_single_with_label(rotation_te_transforms(image), label))
	inputs = torch.stack(inputs)
	inputs, labels = inputs.cuda(), labels.cuda()
	with torch.no_grad():
		outputs = model(inputs.cuda())
		_, predicted = outputs.max(1)
	return predicted.eq(labels).cpu()

def adapt_single(model, image, optimizer, criterion, niter, batch_size):
	model.train()
	for iteration in range(niter):
		inputs = [rotation_tr_transforms(image) for _ in range(batch_size)]
		inputs, labels = rotate_batch(inputs)
		inputs, labels = inputs.cuda(), labels.cuda()
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

def test_single(model, image, label):
	model.eval()
	inputs = te_transforms(image).unsqueeze(0)
	with torch.no_grad():
		outputs = model(inputs.cuda())
		_, predicted = outputs.max(1)
		confidence = nn.functional.softmax(outputs, dim=1).squeeze()[label].item()
	correctness = 1 if predicted.item() == label else 0
	return correctness, confidence
	