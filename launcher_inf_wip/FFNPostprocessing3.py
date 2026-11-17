###
###
###
import sys, os, time, errno, glob
import numpy as np
# import copy
# from itertools import chain
# import subprocess as s
# import threading
# import miscellaneous.Miscellaneous as m
import cv2
import subprocess
import PIL
import PIL.Image


from os import path, pardir
#main_dir = path.abspath(path.dirname(sys.argv[0]))  # Dir of main
#main_dir    = '/home/urakubo/UNI-EM'
#_3D_FFN_dir = '/home/urakubo/UNI-EM/segment/_3D_FFN/ffn'
#sys.path.append(main_dir)
#icon_dir = path.join(main_dir, "icons")
#sys.path.append(path.join(main_dir, "segment"))
#sys.path.append(path.join(main_dir, "system"))

def gen_col_pil(id_data, colordata):
	ncol = id_data.shape[0]
	nrow = id_data.shape[1]
	imgArray = np.zeros((ncol, nrow, 3), dtype='uint8')
	for i in range(ncol):
		for j in range(nrow):
			imgArray[i, j][0] = colordata[id_data[i, j], 0]  # R
			imgArray[i, j][1] = colordata[id_data[i, j], 1]  # G
			imgArray[i, j][2] = colordata[id_data[i, j], 2]  # B
	pilOUT = PIL.Image.fromarray(np.uint8(imgArray))
	return pilOUT

def save_tifc(id_data, filename, colordata):
	pilOUT = gen_col_pil(id_data, colordata)
	pilOUT.save(filename)

def save_pngc(id_data, filename, colordata):
	pilOUT = gen_col_pil(id_data, colordata)
	pilOUT.save(filename)

def fill_holes(data):
	uniq, count = np.unique(data, return_counts=True)
	sorted_uniq  =  uniq[np.argsort(count)]
	sorted_count = count[np.argsort(count)]
	print('Number of labels for fill_holes: {}'.format(sorted_uniq.shape[0]))
	'''
	import fill_voids
	for i, id in enumerate(sorted_uniq):
		print(i, end=' ')
		if id == 0:
			continue
		tmp = fill_voids.fill(data == id, in_place=False)
		data[tmp] = id
	'''
	return data

class FFNPostprocessing():
	def _Run(self, params):
		##
		print('')
		print('Start postprocesing.')
		print('')

		#target_inference_file = os.path.join( params['FFNs Folder'], "0", "0", "seg-0_0_0.npz")
		target_inference_file = \
			os.path.join( params['FFNs Inference Folder'], str(params['Start X']), str(params['Start Y']), \
			"seg-{}_{}_{}.npz".format(params['Start X'], params['Start Y'], params['Start Z']))
		if not os.path.isfile( target_inference_file ):
			print('No inference file found at: ', target_inference_file)
			print('Aborted')
			print('')
			return False
		    
		data = np.load( target_inference_file )
		print('File contents :', data.files)
		segmentation = data['segmentation']
		print('Segmentation image size: ', segmentation.shape)
		filetype = params['Output Filetype']
		##
		if params['Fill holes'] == True:
			print('Fill holes start.')
			#segmentation = fill_holes(segmentation)
			import fastmorph
			segmentation = fastmorph.fill_holes(
				segmentation,
				fix_borders=True,
				morphological_closing=True,
                                remove_enclosed=True)
			print('Fill holes done.')

		##
		if (filetype == '8-bit color PNG') or (filetype == '8-bit color TIFF'):
			ids = np.max(segmentation)
			print('Max segmentation ID: ', ids)
			colormap = np.random.randint(255, size=(ids+2, 3), dtype=np.uint64)
			colormap[0,:] = 0
		##
		for idz in range(segmentation.shape[0]):
			image2d = segmentation[idz, :, :]
			# print('image2d size: ', image2d.shape)

			if params['Output Filetype'] == '16-bit gray scale TIFF':
				filename = os.path.join(params['Output Segmentation Folder'], 'z{:0=4}.tif'.format(idz))		
				cv2.imwrite(filename, image2d.astype('uint16'))
			elif params['Output Filetype'] == '8-bit color PNG' :
				filename = os.path.join(params['Output Segmentation Folder'], 'z{:0=4}.png'.format(idz))
				save_pngc(image2d, filename, colormap)
			elif params['Output Filetype'] == '8-bit color TIFF' :
				filename = os.path.join(params['Output Segmentation Folder'], 'z{:0=4}.tif'.format(idz))
				save_tifc(image2d, filename, colormap)
			else:
				raise ValueError("params['Output Filetype'] was not implemented: ", params['Output Filetype'])
				
			print(filename)
			
		##
		return 0
		##

	def __init__(self):
		##
		pass

