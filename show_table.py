import sys
import numpy as np
import torch
from utils.misc import *
from test_calls.show_result import get_err_adapted

corruptions_names = ['gauss', 'shot', 'impulse', 'defocus', 'glass', 'motion', 'zoom', 
							'snow', 'frost', 'fog', 'bright', 'contra', 'elastic', 'pixel', 'jpeg']
corruptions_names.insert(0, 'orig')

corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
					'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
					'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
corruptions.insert(0, 'original')

info = []
info.append(('gn', '', 5))
########################################################################

def print_table(table, prec1=True):
	for row in table:
		row_str = ''
		for entry in row:
			if prec1:
				row_str += '%.1f\t' %(entry)
			else:
				row_str += '%s\t' %(str(entry))
		print(row_str)

def show_table(folder, level):
	results = []
	for corruption in corruptions:
		row = []
		try:
			rdict_ada = torch.load(folder + '/%s_%d_ada.pth' %(corruption, level))
			rdict_inl = torch.load(folder + '/%s_%d_inl.pth' %(corruption, level))
			row.append(1 - rdict_inl['cls_initial'])
			row.append(1 - rdict_ada['cls_adapted'])
		except:
			row.append(0)
			row.append(0)
		results.append(row)

	results = np.asarray(results)
	results = np.transpose(results)
	results = results * 100
	return results

def show_none(folder, level):
	results = []
	for corruption in corruptions:
		try:
			rdict_inl = torch.load(folder + '/%s_%d_none.pth' %(corruption, level))
			results.append(1 - rdict_inl['cls_initial'])
		except:
			results.append(0)
	results = np.asarray([results])
	results = results * 100
	return results

for parta, partb, level in info:
	print(level, parta + partb)
	print_table([corruptions_names], prec1=False)

	results_none = show_none('results/test_none_%s_%s' %('none', parta), level)
	print_table(results_none)

	results_slow = show_table('results/test_layer3_%s_%s%s' %('slow', parta, partb), level)

	results_onln = show_table('results/test_layer3_%s_%s%s' %('online_shuffle', parta, partb), level)

	results_slow[0,:] = results_onln[0,:]
	results_onln = results_onln[1:,:]

	print_table(results_slow)
	print_table(results_onln)

	results = np.concatenate((results_none, results_slow, results_onln))	
	torch.save(results, 'results/test_layer3_%d_%s%s.pth' %(level, parta, partb))
