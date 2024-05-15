import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from .models import FFDNet
from .utils import normalize,\
				variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb

from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse


denoising_router = APIRouter(prefix='/denoising')


@denoising_router.post('/')
async def denoising_handler(file: UploadFile):
    

	path_to_file = f'./temp/{file.filename}'
	with open(path_to_file, 'wb+') as file_obj:
		file_obj.write(file.file.read())
	try:
		rgb_den = is_rgb(path_to_file)
	except:
		raise Exception('Could not open the input image')

	if rgb_den:
		in_ch = 3
		model_fn = './source/denoising_model/net_rgb.pth'
		imorig = cv2.imread(path_to_file)
		imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
	else:
		in_ch = 1
		model_fn = './source/denoising_model/net_gray.pth'
		imorig = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
		imorig = np.expand_dims(imorig, 0)
	imorig = np.expand_dims(imorig, 0)

	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = imorig.shape
	if sh_im[2]%2 == 1:
		expanded_h = True
		imorig = np.concatenate((imorig, \
				imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

	if sh_im[3]%2 == 1:
		expanded_w = True
		imorig = np.concatenate((imorig, \
				imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

	imorig = normalize(imorig)
	imorig = torch.Tensor(imorig)


	net = FFDNet(num_input_channels=in_ch)

	state_dict = torch.load(model_fn, map_location='cpu')
	state_dict = remove_dataparallel_wrapper(state_dict)
	model = net
	model.load_state_dict(state_dict)

	model.eval()

	
	dtype = torch.FloatTensor

	
	imnoisy = imorig.clone()

        # Test mode
	with torch.no_grad(): # PyTorch v0.4.0
		imorig, imnoisy = Variable(imorig.type(dtype)), Variable(imnoisy.type(dtype))
		nsigma = Variable(torch.FloatTensor([1]).type(dtype))

	# Measure runtime

	# Estimate noise and subtract it to the input image
	im_noise_estim = model(imnoisy, nsigma)
	outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)

	if expanded_h:
		imorig = imorig[:, :, :-1, :]
		outim = outim[:, :, :-1, :]
		imnoisy = imnoisy[:, :, :-1, :]

	if expanded_w:
		imorig = imorig[:, :, :, :-1]
		outim = outim[:, :, :, :-1]
		imnoisy = imnoisy[:, :, :, :-1]

	# Compute PSNR and log it

	# Compute difference
	diffout   = 2*(outim - imorig) + .5
	diffnoise = 2*(imnoisy-imorig) + .5

	# Save images
	noisyimg = variable_to_cv2_image(imnoisy)
	outimg = variable_to_cv2_image(outim)
	cv2.imwrite(path_to_file, outimg)
		

	return FileResponse(path_to_file)

