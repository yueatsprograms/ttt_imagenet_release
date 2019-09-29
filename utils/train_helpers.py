import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data

from utils.rotation import RotateImageFolder
from models.SSHead import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tr_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									normalize])
te_transforms = transforms.Compose([transforms.Resize(256),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize])

rotation_tr_transforms = tr_transforms
rotation_te_transforms = te_transforms

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
	                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
	                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def build_model(args):
	if args.group_norm == 0:
		norm_layer = None
	else:
		def gn_helper(planes):
			return nn.GroupNorm(args.group_norm, planes)
		norm_layer = gn_helper

	width = 1
	if args.depth == 152:
		net = models.resnet152(norm_layer=norm_layer).cuda()
		expansion = 4
	elif args.depth == 50:
		net = models.resnet50(norm_layer=norm_layer).cuda()
		expansion = 4
	elif args.depth == 18:
		net = models.resnet18(norm_layer=norm_layer).cuda()
		expansion = 1

	planes = 512
	if args.shared == 'none':
		args.shared = None
	if args.shared == 'layer4' or args.shared is None:
		ext = extractor_from_layer4(net)
		head = nn.Linear(expansion * planes, 4)
	elif args.shared == 'layer3':
		ext = extractor_from_layer3(net)
		head = copy.deepcopy([net.layer4, net.avgpool, 
								ViewFlatten(), nn.Linear(expansion * planes * width, 4)])
		head = nn.Sequential(*head)
	elif args.shared == 'layer2':
		ext = extractor_from_layer2(net)
		head = copy.deepcopy([net.layer3, net.layer4, net.avgpool, 
								ViewFlatten(), nn.Linear(expansion * planes * width, 4)])
		head = nn.Sequential(*head)

	ssh = ExtractorHead(ext, head).cuda()
	net = torch.nn.DataParallel(net)
	ssh = torch.nn.DataParallel(ssh)
	return net, ext, head, ssh

class ImagePathFolder(datasets.ImageFolder):
	def __init__(self, traindir, train_transform):
		super(ImagePathFolder, self).__init__(traindir, train_transform)	

	def __getitem__(self, index):
		path, _ = self.imgs[index]
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)
		path, pa = os.path.split(path)
		path, pb = os.path.split(path)
		return img, 'val/%s/%s' %(pb, pa)

def prepare_train_data(args):
	print('Preparing data...')
	traindir = os.path.join(args.dataroot, 'train')
	trset = RotateImageFolder(traindir, tr_transforms, original=True, rotation=args.rotation,
														rotation_transform=rotation_tr_transforms)
	trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size, shuffle=True,
													num_workers=args.workers, pin_memory=True)
	return trset, trloader

def prepare_test_data(args, use_transforms=True):
	te_transforms_local = te_transforms if use_transforms else None	
	if not hasattr(args, 'corruption') or args.corruption == 'original':
		print('Test on the original test set')
		validdir = os.path.join(args.dataroot, 'val')
		teset = RotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
													rotation_transform=rotation_te_transforms)

	elif args.corruption in common_corruptions:
		print('Test on %s level %d' %(args.corruption, args.level))
		validdir = os.path.join(args.dataroot, 'imagenet-c', args.corruption, str(args.level))
		teset = RotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
													rotation_transform=rotation_te_transforms)

	elif args.corruption == 'video':
		validdir = os.path.join(args.dataroot, 'val')
		teset = ImagePathFolder(validdir, te_transforms_local)
	else:
		raise Exception('Corruption not found!')
		
	if not hasattr(args, 'workers'):
		args.workers = 1
	teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=False,
													num_workers=args.workers, pin_memory=True)
	return teset, teloader
    
def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def plot_epochs(all_err_cls, all_err_ssh, fname):
	import matplotlib.pyplot as plt
	plt.plot(np.asarray(all_err_cls)*100, color='r', label='supervised')
	plt.plot(np.asarray(all_err_ssh)*100, color='b', label='self-supervised')
	plt.xlabel('epoch')
	plt.ylabel('test error (%)')
	plt.legend()
	plt.savefig(fname)
	plt.close()
