#
# Conduct FFN inference through CUI.
#
# Usage:
# python3 main_inf.py
# python3 main_inf.py
#

import sys, os, time, errno, glob
import numpy as np
from datetime import datetime as dt
import pprint
import cv2
import subprocess
import h5py


from os import path, pardir
dir_base = path.abspath(path.dirname(sys.argv[0]))



# from system.Params import Params
from src.FFNInference3 import FFNInference
from src.FFNPostprocessing3 import FFNPostprocessing



def set_params():

	# Took: 00:XX:XX:XX for 164 x 1024 x 1024

	# Took: 00:02:05:04 for 256 x 1024 x 1024
	# Took: 00:04:03:35 for 352 x 1024 x 1024, Max segmentation ID:  6731
	# Took: 00:12:29:30 for 640 x 1024 x 1024, Max segmentation ID:  11581

	params = {}


	dir_target = 'test_all_z_slices'
	dir_output = 'test_all_z_slices'
	params['Model Folder'] = "trained_models"
	params['ckpt'] = '38420160'

	dir_target = 'test_352_slices'
	dir_output = 'test_352_slices_sparse_0.012b'
	params['Model Folder'] = "trained_models_sparse"
	params['ckpt'] = '12053618'

	dir_target = 'test'
	dir_output = 'test_0.11b'
	params['Model Folder'] = "trained_models"
	params['ckpt'] = '101103410'

			
		
	dir_target = 'test'
	dir_output = 'test_sparse_0.200b'
	params['Model Folder'] = "trained_models_sparse"
	params['ckpt'] = '199984998'
        
	dir_target = 'test'
	dir_output = 'test_sparse_rot_0.200b'
	params['Model Folder'] = "trained_models_sparse_rot"
	params['ckpt'] = '199936740'

	dir_target = 'test'
	dir_output = 'test_sparse_no_myelin_0.184b'
	params['Model Folder'] = "trained_models_sparse_no_myelin"
	params['ckpt'] = '183777070'

	dir_target = 'test'
	dir_output = 'test_sparse_no_myelin_rot_0.105b'
	params['Model Folder'] = "trained_models_sparse_no_myelin_rot"
	params['ckpt'] = '105683619'

	dir_target = 'test'
	dir_output = 'test_sparse_no_myelin_0.200b_20000'
	params['Model Folder'] = "trained_models_sparse_no_myelin"
	params['ckpt'] = '200000001'


	dir_target = 'test_images_for_gt_cell_bound'
	dir_output = 'test_images_for_gt_cell_bound'
	params['Model Folder'] = "trained_models_sparse_no_myelin"
	params['ckpt'] = '200000001'


	os.environ['CUDA_VISIBLE_DEVICES'] = '3'

	'''		
	params['Target Image Folder']        = os.path.join('..', 'test_images',dir_target)
	params['FFNs Inference Folder']      = os.path.join('inferred_segs', dir_output)
	params['Output Segmentation Folder'] = os.path.join('inferred_segs', dir_output)
	params['target_image_file_h5']       = os.path.join(params['FFNs Inference Folder'], 'grayscale_inf.h5')
	'''
    
	params['Target Image Folder']        = os.path.join('..', 'test_images',dir_target)
	params['FFNs Inference Folder']      = os.path.join('/home2/urakubo/seg_schmidt2025/ffn/test_inf1', dir_output)
	params['Output Segmentation Folder'] = os.path.join('/home2/urakubo/seg_schmidt2025/ffn/test_inf1', dir_output)
	params['target_image_file_h5']       = os.path.join(params['FFNs Inference Folder'], 'grayscale_inf.h5')



	print('Target {} '.format(dir_target))

	'''
	params['Start X'] = 0
	params['Start Y'] = 0
	params['Start Z'] = 0
	params['End X'] = 2048
	params['End Y'] = 2048
	params['End Z'] = 268
	if int(row) == 0:
		params['Start Y'] = 256
	if int(column) == 0:
		params['Start X'] = 512+64
	elif int(column) == 6:
		params['End X'] = 2048 - 256
	'''
	
	params['Fill holes'] = True
	params['Sparse Z'] = True
	params['Checkpoint Interval'] = 1800
	#params['Output Filetype'] = "16-bit gray scale TIFF" # "8-bit color PNG", "16-bit gray scale TIFF"
	params['Output Filetype'] = "8-bit color PNG"
	params['dir_repo'] = "repo"
	
	params['Invert'] = False
	if params['Invert'] == True:
		params['Forward Segmentation Folder'] = \
		    os.path.join('inferred_segs', dir_target,\
		    str(params['Start X']), str(params['Start Y']) )

	
	os.makedirs(params['FFNs Inference Folder'], exist_ok=True)
	os.makedirs(params['Output Segmentation Folder'], exist_ok=True)
	
	pprint.pprint(params)
	savefile = os.path.join(params['Output Segmentation Folder'], 'params.txt') 
	with open(savefile, "w") as f:
		pprint.pprint(params, stream=f)

	return params



def prepare_target_image(params):
	files_em = glob.glob( os.path.join(params['Target Image Folder'], "*") ) 
	files_em = sorted(files_em)
	images = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in files_em ]
	images = np.array(images)
	if {"Start Z", "Start Y", "Start X", "End Z",  "End Y",  "End X"} <= set(params.keys()):
		images = images[params['Start Z']:params['End Z'], \
		                  params['Start Y']:params['End Y'], \
		                  params['Start X']:params['End X']]

	images_shape = images.shape
	params["Start Z"] = 0
	params["Start Y"] = 0
	params["Start X"] = 0
	params["End Z"]   = images_shape[0]
	params["End Y"]   = images_shape[1]
	params["End X"]   = images_shape[2]

	print('Processed image shape : ', images.shape)
	params['image_mean'] = np.mean(images).astype(np.int16)
	params['image_std']  = np.std(images).astype(np.int16)

	#
	print('')
	with h5py.File( params['target_image_file_h5'], 'w') as f:
		f.create_dataset('raw', data=images, compression='gzip')
	print('"grayscale_inf.h5" file (target inference image) was generated.')
	print('')
	return params



if __name__ == '__main__':
	'''	
	Usage:
	python3 main_inf_post.py
	'''

	params = set_params()
	params = prepare_target_image(params)
	
	time_start = dt.now()
	
	flag = FFNInference()._Run(params)
	flag = FFNPostprocessing()._Run(params)
	
	time_end = dt.now()
	elapsed = time_end - time_start
	print("Took: %02d:%02d:%02d:%02d" % (elapsed.days, elapsed.seconds // 3600, elapsed.seconds // 60 % 60, elapsed.seconds % 60))
	print('Target {}, done.'.format(params['Target Image Folder']))


    
