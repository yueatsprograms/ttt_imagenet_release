from __future__ import print_function
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.folder import default_loader as loader

from tqdm import tqdm
from utils.misc import *
from utils.test_helpers import test
from utils.adapt_helpers import *
from utils.imagenet_vid import convert_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/data/datasets/imagenet/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--batch_size', default=32, type=int)
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument('--online', action='store_true')
########################################################################
parser.add_argument('--load_results', action='store_true')
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='.')

args = parser.parse_args()
args.threshold += 0.001		# to correct for numeric errors
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)

print('Resuming from %s...' %(args.resume))
ckpt = torch.load('%s/ckpt.pth' %(args.resume))
net.load_state_dict(ckpt['net'])
head.load_state_dict(ckpt['head'])

args.corruption = 'video'
args.dataroot += 'imagenet-vid-robust-example/imagenet-vid-robust/'
with open(args.dataroot + "/metadata/pmsets.json", "r") as f:
	pmsets = json.loads(f.read())
with open(args.dataroot + "/metadata/labels.json" , "r") as f:
	labels = json.loads(f.read())

with open(args.dataroot + "/misc/wnid_map.json", "r") as f:
	wnid = json.loads(f.read())
with open(args.dataroot + "/misc/imagenet_class_index.json", "r") as f:
	ia = json.loads(f.read())
with open(args.dataroot + "/misc/imagenet_vid_class_index.json", "r") as f:
	ib = json.loads(f.read())

def run_plain():
	preds = {}
	net.eval()
	_, teloader = prepare_test_data(args)
	for i, (inputs, files) in tqdm(enumerate(teloader)):
		with torch.no_grad():
			inputs = inputs.cuda()
			outputs = net(inputs)
			outputs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
			outputs = convert_predictions(outputs, ia, ib, wnid)
		for output, file in zip(outputs, files):
			preds[file] = output
	return preds

def pred_single(model, image):
	model.eval()
	inputs = te_transforms(image).unsqueeze(0)
	with torch.no_grad():
		outputs = model(inputs.cuda())
		outputs = nn.functional.softmax(outputs, dim=1).cpu().numpy()		
	return convert_predictions(outputs, ia, ib, wnid)

def run_adapt():
	preds = {}
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = optim.SGD(ssh.parameters(), lr=args.lr)

	def local_adapt(name, sshconf):
		image = loader(args.dataroot + name)
		sshconf.append(test_single(ssh, image, 0)[1])
		if sshconf[-1] < args.threshold:
			adapt_single(ssh, image, optimizer, criterion, args.niter, args.batch_size)
		preds[name] = pred_single(net, image)

	for anchor, pmset in tqdm(pmsets.items()):
		if not args.online:
			net.load_state_dict(ckpt['net'])
			head.load_state_dict(ckpt['head'])

		sshconf = []
		local_adapt(anchor, sshconf)
		# for elem in pmset:
		# 	local_adapt(elem, sshconf)	
	return preds

def pmk_from_preds(preds):
	one_hot_anc = []
	one_hot_pmk = []
	correct_pmk = []
	for anchor, pmset in pmsets.items():
		correct_anc = np.argmax(preds[anchor]) in labels[anchor]
		correct_pms = [correct_anc]
		one_hot_anc.append(correct_anc)

		# for elem in pmset:
		# 	correct_pms.append(np.argmax(preds[elem]) in labels[elem])
		correct_pmk.append(correct_pms)
		one_hot_pmk.append(all(correct_pms))
	return one_hot_anc, one_hot_pmk, correct_pmk


if args.load_results:
	preds_plain = torch.load(args.outf + '/plain.pth')
else:
	preds_plain = run_plain()
	torch.save(preds_plain, args.outf + '/plain.pth')

one_hot_anc, _, _ = pmk_from_preds(preds_plain)
print('Old accuracy: %.2f' %(mean(one_hot_anc)*100))

if args.load_results:
	preds_adapt = torch.load(args.outf + '/adapt.pth')
else:
	preds_adapt = run_adapt()
	torch.save(preds_adapt, args.outf + '/adapt.pth')

one_hot_anc, _, _ = pmk_from_preds(preds_adapt)
print('New accuracy: %.2f' %(mean(one_hot_anc)*100))
