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


denoising_router = APIRouter(prefix='/denoising') #Создание API-маршрутизатора для обработки запросов по удалению шума:


@denoising_router.post('/')			# Обработчик для POST-запросов по пути '/':
async def denoising_handler(file: UploadFile):
    

	path_to_file = f'./temp/{file.filename}'		#Запись загруженного файла во временный файл:
	with open(path_to_file, 'wb+') as file_obj:
		file_obj.write(file.file.read())
	try:											#Проверка формата изображения (RGB или оттенки серого):
		rgb_den = is_rgb(path_to_file)
	except:
		raise Exception('Could not open the input image')

	if rgb_den:										#Чтение изображения в соответствующем формате и подготовка его для обработки:
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

	# Handle odd sizes						#Обработка изображения: подготовка к обработке нечетных размеров, нормализация и преобразование в тензор:
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


	net = FFDNet(num_input_channels=in_ch)		# Создание модели для удаления шума и загрузка предварительно обученных весов

	# Load saved weights
	
	#state_dict = torch.load(model_fn, map_location='cpu') 
	# state_dict = remove_dataparallel_wrapper(state_dict)
	# model = net

	state_dict = torch.load(model_fn) 
	device_ids = [0]
	model = nn.DataParallel(net, device_ids=device_ids).cuda()

	model.load_state_dict(state_dict)

	model.eval()		# Переключение модели в режим оценки:


	# Sets data type according to CPU or GPU modes
	dtype = torch.cuda.FloatTensor
	# dtype = torch.FloatTensor		#Подготовка данных для обработки (тензоры):

	imnoisy = imorig.clone()

    # Test mode           	# Запуск процесса удаления шума с помощью модели:
	with torch.no_grad(): 
		imorig, imnoisy = Variable(imorig.type(dtype)), Variable(imnoisy.type(dtype))
		nsigma = Variable(torch.FloatTensor([1]).type(dtype))

	# Estimate noise and subtract it to the input image
	im_noise_estim = model(imnoisy, nsigma)
	outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)

	if expanded_h:				# Обрезка изображений до исходного размера, если они были расширены:
		imorig = imorig[:, :, :-1, :]
		outim = outim[:, :, :-1, :]
		imnoisy = imnoisy[:, :, :-1, :]

	if expanded_w:
		imorig = imorig[:, :, :, :-1]
		outim = outim[:, :, :, :-1]
		imnoisy = imnoisy[:, :, :, :-1]

	# Compute difference 				#Вычисление различий между изображениями (оригинальным, зашумленным и обработанным):
	diffout   = 2*(outim - imorig) + .5
	diffnoise = 2*(imnoisy-imorig) + .5

	# Save images 					# Преобразование тензоров обратно в изображения OpenCV и сохранение результата обработки во временный файл:
	noisyimg = variable_to_cv2_image(imnoisy)
	outimg = variable_to_cv2_image(outim)
	cv2.imwrite(path_to_file, outimg)
		

	return FileResponse(path_to_file)	#Возвращение обработанного изображения в ответ на запрос:

