###
###
###
import sys, os, time, errno, glob
import numpy as np
import subprocess as s
import fnmatch
import cv2
import h5py

import contextlib

from os import path, pardir
from pathlib import Path

##
##
##

class FFNInference():

	def write_text(self, f, key, value, t):
		if key == 'min_boundary_dist':
			f.write('{0}{1} {2} \n'.format(t, key, value ) )
		elif isinstance(value, str):
			f.write('{0}{1}: "{2}" \n'.format(t, key, value ) )
		else:
			f.write('{0}{1}: {2} \n'.format(t, key, value))

	def write_call(self, f, request, t):
		for key, value in request.items():
			if isinstance(request[key], dict):
				f.write('{} {{\n'.format(key))
				tid = t + "  "
				self.write_call(f, request[key], tid)
				f.write('}\n')
			else:
				self.write_text(f, key, value, t)

	def _Run(self, params):

		# self.u_info = u_info
		##
		## Remove preovious results.
		##
		removal_file1 = os.path.join( params['FFNs Inference Folder'], str(params['Start X']), str(params['Start Y']), \
			"seg-{}_{}_{}.npz".format(params['Start X'], params['Start Y'], params['Start Z']))
		removal_file2 = os.path.join( params['FFNs Inference Folder'], str(params['Start X']), str(params['Start Y']), \
			"seg-{}_{}_{}.prob".format(params['Start X'], params['Start Y'], params['Start Z']))
		#removal_file1 = os.path.join( params['FFNs Folder'] ,'0','0','seg-0_0_0.npz' )
		#removal_file2 = os.path.join( params['FFNs Folder'], '0','0','seg-0_0_0.prob')

		if os.path.isfile(removal_file1) or os.path.isfile(removal_file2) :
			question = "Previous result of inference has been found in the FFNs Folder. Remove them?"
			reply = self.query_yes_no(question, default="yes")

			if reply == True:
				with contextlib.suppress(FileNotFoundError):
					    os.remove(removal_file1)
				with contextlib.suppress(FileNotFoundError):
					    os.remove(removal_file2)
				print('Inference files were removed.')
			else:
				print('FFN inference was canceled.')
				#m.LockFolder(parent.u_info,  params['FFNs Folder'])
				return

		##
		## Tensorflow model extracted
		##

		if 'ckpt' in params.keys():
			max_id_model = os.path.join(params['Model Folder'], 'model.ckpt-' + str(params['ckpt'])  )
		else:
			max_id_model = self.SelectMaxModel(params['Model Folder'] )
			if max_id_model == False:
				print('Cannot find tensorflow model.')
				return False
		print( 'Tensorflow model : ', max_id_model )

		##
		## Inference configration file generation
		##

		request = {}
		request['image'] = {"hdf5": "{}:raw".format(params['target_image_file_h5']).replace('\\', '/') }
		request['image_mean']   = params['image_mean']
		request['image_stddev'] = params['image_std']
		request['checkpoint_interval'] = int(params['Checkpoint Interval'])

		if ('Invert' in params.keys()) and (params['Invert'] == True):
			request['seed_policy'] = "PolicyInvertOrigins"
			request['seed_policy_args'] = '{\\\"segmentation_dir\\\": \\\"'+params['Forward Segmentation Folder']+'\\\"}'
		else:
			request['seed_policy'] = "PolicySparsePeaks"

		request['model_checkpoint_path'] = max_id_model.replace('\\', '/')
		request['model_name'] = "convstack_3d.ConvStack3DFFNModel"

		if params['Sparse Z'] == True:
			request['model_args'] = "{\\\"depth\\\": 9, \\\"fov_size\\\": [33, 33, 17], \\\"deltas\\\": [8, 8, 4]}"
		else :
			request['model_args'] = "{\\\"depth\\\": 12, \\\"fov_size\\\": [33, 33, 33], \\\"deltas\\\": [8, 8, 8]}"

		request['segmentation_output_dir'] = params['FFNs Inference Folder'].replace('\\', '/')
		inference_options = {}
		inference_options['init_activation'] = 0.95
		inference_options['pad_value'] = 0.05 # 0.05 old 0.5 New
		inference_options['move_threshold'] = 0.9
		inference_options['min_boundary_dist'] = "{x: 1, y: 1, z: 1}" # {"x": 1, "y": 1, "z": 1} # 1,1,1 old 2,2,1 New
		inference_options['segment_threshold'] = 0.6
		inference_options['min_segment_size'] = 4000
		request['inference_options'] = inference_options


		config_file = os.path.join(params['FFNs Inference Folder'], "inference_params.pbtxt")
		with open(config_file, "w", encoding='utf-8') as f:
			self.write_call(f, request, "")

		print('')
		print('Configuration file was saved at :')
		print(config_file)
		print('')
		##
		## Inference start (I gave up the use of run_inference because of the augment parsing problem)
		##
		os.makedirs(os.path.join( params['FFNs Inference Folder'] ,str(params['Start X']), str(params['Start Y']) ) , exist_ok=True)
		##
		#python_exe = sys.executable
		python_exe = 'python'
		
		## For view
		comm_inference = [python_exe, os.path.join(params['dir_repo'], 'run_inference.py')]
		bounding_box = "'start {{ x:{} y:{} z:{} }} size {{ x:{} y:{} z:{} }}'".\
		    format(params['Start X'], params['Start Y'], params['Start Z'], \
		           params['End X']-params['Start X'], params['End Y']-params['Start Y'], params['End Z']-params['Start Z'])
		arguments_view = ['--inference_request="$(cat '+config_file+')"',
			'--bounding_box', bounding_box
		        ]
		comm_inference_view = comm_inference + arguments_view
				
		## For run
		#with open(config_file, "r", encoding='utf-8') as f:
		#	configs = f.read()
		configs = Path(config_file).read_text()
		bounding_box = "start {{ x:{} y:{} z:{} }} size {{ x:{} y:{} z:{} }}".\
		    format(params['Start X'], params['Start Y'], params['Start Z'], \
		           params['End X']-params['Start X'], params['End Y']-params['Start Y'], params['End Z']-params['Start Z'])
		arguments = ["--inference_request", configs,
			'--bounding_box', bounding_box
		        ]
		comm_inference = comm_inference + arguments
		
		
		print(' '.join(comm_inference_view))
		print('')
		s.run(comm_inference)
		print('')
		print('Inference was finished.')
		print('')
		return True
		##


	def SelectMaxModel(self, folder_path):

		required_files = [ \
			'model.ckpt-*.meta', \
			'model.ckpt-*.index',\
			'model.ckpt-*.data-00000-of-00001' ]
		tmp = glob.glob( path.join(folder_path, "*") )
		filenames_in_folder = [os.path.basename(r) for r in tmp]
		cropped = []
		for required_file in required_files:
			tmp = fnmatch.filter(filenames_in_folder, required_file)
			if len(tmp) == 0:
				return False
			a, b = map(len, required_file.split('*'))
			cropped.append( {t[a:-b] for t in tmp} ) 
		intersection = cropped[0] & cropped[1] & cropped[2] 
		if len(intersection) == 0:
			return False
		max_id_name = os.path.join(folder_path, 'model.ckpt-' + str(max(map(int, intersection)))  )
		return max_id_name

	def query_yes_no(self, question, default="yes"):

		valid = {"yes":True, "y":True, "ye":True, "no":False, "n":False}
		if default == None:
			prompt = " [y/n] "
		elif default == "yes":
			prompt = " [Y/n] "
		elif default == "no":
			prompt = " [y/N] "
		else:
			raise ValueError("invalid default answer: '%s'" % default)

		while True:
			sys.stdout.write(question + prompt)
			choice = input().lower()
			if default is not None and choice == '':
				return valid[default]
			elif choice in valid:
				return valid[choice]
			else:
				sys.stdout.write("Please respond with 'yes' or 'no' "\
						 "(or 'y' or 'n').\n")
	# Usage example

	def __init__(self):
	        pass
	        
