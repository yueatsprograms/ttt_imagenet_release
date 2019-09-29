import argparse
import numpy as np
import torch
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--outf', default='.')

args = parser.parse_args()
rdict = torch.load(args.outf + '/%s_%d_inl.pth' %(args.corruption, args.level))
fname = args.outf + '/%s_%d' %(args.corruption, args.level)

def plot_losses(cls_losses, ssh_losses, fname):
	import matplotlib.pyplot as plt
	from utils.misc import normalize

	cls_losses = normalize(cls_losses)
	ssh_losses = normalize(ssh_losses)
	correlation = pearsonr(cls_losses, ssh_losses)
	print('correlation: %.3f, significance: %.3f' %(correlation[0], correlation[1]))
	plt.scatter(cls_losses, ssh_losses, color='r', s=4)
	plt.xlabel('supervised loss')
	plt.ylabel('self-supervised loss')
	plt.savefig('%s_scatter.pdf' %(fname))
	plt.close()

def decomp_rand(clse, sshe, total):
	clsw = total * clse
	clsr = total - clsw

	crr = clsr * (1-sshe)
	crw = clsr * sshe
	cwr = clsw * (1-sshe)
	cww = clsw * sshe
	return int(crr), int(crw), int(cwr), int(cww)

def show_decomp(cls_initial, cls_correct, ssh_initial, ssh_correct, fname):
	import matplotlib.pyplot as plt
	from utils.test_helpers import count_each, pair_buckets

	dtrue = count_each(pair_buckets(cls_correct, ssh_correct))
	torch.save(dtrue, '%s_dec.pth' %(fname))
	print('Error decoposition:', *dtrue)
	drand = decomp_rand(cls_initial, ssh_initial, sum(dtrue))	

	width = 0.25
	ind = np.arange(4)
	plt.bar(ind, 		drand, width, label='independent')
	plt.bar(ind+width, 	dtrue, width, label='observed')

	plt.ylabel('count')
	plt.xticks(ind + width/2, ('RR', 'RW', 'WR', 'WW'))
	plt.legend(loc='best')
	plt.savefig('%s_bar.pdf' %(fname))
	plt.close()

plot_losses(rdict['cls_losses'], rdict['ssh_losses'], fname)
show_decomp(rdict['cls_initial'], rdict['cls_correct'],
			rdict['ssh_initial'], rdict['ssh_correct'], fname)
