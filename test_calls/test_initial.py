from __future__ import print_function
import argparse
import torch

from utils.misc import *
from utils.test_helpers import test
from utils.train_helpers import *

parser = argparse.ArgumentParser()
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--dataroot', default='/data/datasets/imagenet/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--workers', default=8, type=int)
########################################################################
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='.')
parser.add_argument('--none', action='store_true')

args = parser.parse_args()
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)
teset, teloader = prepare_test_data(args)

print('Resuming from %s...' %(args.resume))
ckpt = torch.load('%s/ckpt.pth' %(args.resume))
net.load_state_dict(ckpt['net'])
teloader.dataset.switch_mode(True, False)
cls_initial, cls_correct, cls_losses = test(teloader, net, verbose=True)

print('Old test error cls %.2f' %(ckpt['err_cls']*100))
print('New test error cls %.2f' %(cls_initial*100))

if args.none:
	rdict = {'cls_initial': cls_initial, 'cls_correct': cls_correct, 'cls_losses': cls_losses}
	torch.save(rdict, args.outf + '/%s_%d_none.pth' %(args.corruption, args.level))
	quit()

print('Old test error ssh %.2f' %(ckpt['err_ssh']*100))
head.load_state_dict(ckpt['head'])

teloader.dataset.switch_mode(False, True)
ssh_initial, ssh_correct, ssh_losses = test(teloader, ssh, verbose=True)
print('New test error ssh %.2f' %(ssh_initial*100))

rdict = {'cls_initial': cls_initial, 'cls_correct': cls_correct, 'cls_losses': cls_losses,
		 'ssh_initial': ssh_initial, 'ssh_correct': ssh_correct, 'ssh_losses': ssh_losses}
torch.save(rdict, args.outf + '/%s_%d_inl.pth' %(args.corruption, args.level))
