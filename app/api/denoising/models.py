"""
Definition of the FFDNet model and its custom layers

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch.nn as nn
from torch.autograd import Variable
from .functions import *
class UpSampleFeatures(nn.Module):	#наследует функциональность класса nn.Module из библиотеки PyTorch. Это позволяет нам использовать этот класс в качестве компонента модели и определять его собственные методы и параметры.
	r"""Implements the last layer of FFDNet
	"""
	def __init__(self): 			# Это начало определения конструктора класса. Конструктор выполняет инициализацию объекта класса и его параметров. В данном случае конструктор не принимает дополнительных параметров, поэтому в скобках после self ничего нет.
		super(UpSampleFeatures, self).__init__()		#Эта строка вызывает конструктор родительского класса (в данном случае nn.Module), чтобы убедиться, что все необходимые операции инициализации класса выполняются корректно. Это важно, чтобы правильно настроить класс для использования в модели.
	def forward(self, x):
		return upsamplefeatures(x)

class IntermediateDnCNN(nn.Module):
	r"""Implements the middel part of the FFDNet architecture, which
	is basically a DnCNN net
	"""
	def __init__(self, input_features, middle_features, num_conv_layers):# Это начало определения конструктора класса. Конструктор выполняет инициализацию объекта класса и его параметров. Он принимает три параметра: input_features (количество входных признаков), middle_features (количество признаков в промежуточных слоях) и num_conv_layers (количество сверточных слоев).
		super(IntermediateDnCNN, self).__init__()#Эта строка вызывает конструктор родительского класса (в данном случае nn.Module), чтобы убедиться, что все необходимые операции инициализации класса выполняются корректно. Это важно, чтобы правильно настроить класс для использования в модели.
		self.kernel_size = 3#Эта строка определяет размер ядра свертки, который будет использоваться в сверточных слоях. В данном случае ядро имеет размер 3x3.
		self.padding = 1 #Эта строка определяет количество пикселей, которое будет добавлено к входным данным перед выполнением свертки. Это необходимо для того, чтобы сохранить размеры изображения после применения сверточных операций.
		self.input_features = input_features	#Эти строки просто присваивают переданные аргументы конструктора класса соответствующим атрибутам класса.
		self.num_conv_layers = num_conv_layers
		self.middle_features = middle_features
		if self.input_features == 5:					#Этот блок проверяет количество входных признаков и соответственно определяет количество выходных признаков. В случае градационных изображений (5 входных признаков) будет 4 выходных, а для RGB-изображений (15 входных признаков) будет 12 выходных.
			self.output_features = 4 #Grayscale image
		elif self.input_features == 15:
			self.output_features = 12 #RGB image
		else:
			raise Exception('Invalid number of input features')

		layers = []		#Эта строка создает пустой список, в который будут добавлены слои сети.
		layers.append(nn.Conv2d(in_channels=self.input_features,\
								out_channels=self.middle_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		layers.append(nn.ReLU(inplace=True))
		for _ in range(self.num_conv_layers-2):							#Этот цикл создает указанное количество сверточных слоев, указанных в аргументе num_conv_layers. Каждый сверточный слой включает в себя операции свертки, нормализации и активации.
			layers.append(nn.Conv2d(in_channels=self.middle_features,\
									out_channels=self.middle_features,\
									kernel_size=self.kernel_size,\
									padding=self.padding,\
									bias=False))
			layers.append(nn.BatchNorm2d(self.middle_features))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Conv2d(in_channels=self.middle_features,\
								out_channels=self.output_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		self.itermediate_dncnn = nn.Sequential(*layers)			# Эта строка создает последовательный контейнер nn.Sequential, который объединяет все созданные слои в один последовательный блок.
	def forward(self, x):				#Это определение метода forward, который определяет, как данные будут проходить через слои класса при передаче данных через модель. В данном случае метод forward просто передает входные данные x через созданный последовательный блок слоев и возвращает результат.
		out = self.itermediate_dncnn(x)
		return out

class FFDNet(nn.Module):
	r"""Implements the FFDNet architecture
	"""
	def __init__(self, num_input_channels):		#Это начало определения конструктора класса. Конструктор выполняет инициализацию объекта класса и его параметров. Он принимает один параметр - num_input_channels, который указывает количество каналов во входном изображении.
		super(FFDNet, self).__init__()			#Эта строка вызывает конструктор родительского класса (в данном случае nn.Module), чтобы убедиться, что все необходимые операции инициализации класса выполняются корректно. Это важно для правильной настройки класса для использования в модели.
		self.num_input_channels = num_input_channels
		if self.num_input_channels == 1:		#Этот блок проверяет количество входных каналов и в зависимости от этого определяет параметры модели: количество признаковых карт, количество сверточных слоев и другие.
			# Grayscale image
			self.num_feature_maps = 64
			self.num_conv_layers = 15
			self.downsampled_channels = 5
			self.output_features = 4
		elif self.num_input_channels == 3:
			# RGB image
			self.num_feature_maps = 96
			self.num_conv_layers = 12
			self.downsampled_channels = 15
			self.output_features = 12
		else:
			raise Exception('Invalid number of input features')

		self.intermediate_dncnn = IntermediateDnCNN(\			# Эта строка создает экземпляр класса IntermediateDnCNN, который представляет собой промежуточную часть архитектуры модели FFDNet. В качестве аргументов передаются параметры, определенные в предыдущем блоке условия (if-elif-else). Этот экземпляр будет использоваться для преобразования входных данных в промежуточные признаки.
				input_features=self.downsampled_channels,\
				middle_features=self.num_feature_maps,\
				num_conv_layers=self.num_conv_layers)
		self.upsamplefeatures = UpSampleFeatures()			#Эта строка создает экземпляр класса UpSampleFeatures, который представляет собой последний слой архитектуры модели FFDNet. Этот экземпляр будет использоваться для восстановления размеров признаков до исходных размеров.

##Этот метод forward определяет процесс прямого прохода (forward pass) модели FFDNet. В этом методе происходит последовательная обработка входных данных через промежуточную часть сети IntermediateDnCNN и затем через последний слой UpSampleFeatures

	def forward(self, x, noise_sigma):	#Это определение метода forward для класса FFDNet. Он принимает два параметра: x - входные данные (изображение) и noise_sigma - уровень шума во входном изображении.
		concat_noise_x = concatenate_input_noise_map(\    #Эта строка вызывает функцию concatenate_input_noise_map, которая объединяет входные данные x с уровнем шума noise_sigma. В результате получается тензор, содержащий как изображение, так и информацию о шуме.
				x.data, noise_sigma.data)
		concat_noise_x = Variable(concat_noise_x)		#Эта строка создает переменную PyTorch Variable из тензора concat_noise_x. В PyTorch рекомендуется использовать Variable для отслеживания операций и автоматического вычисления градиентов.
		h_dncnn = self.intermediate_dncnn(concat_noise_x)#Эта строка передает объединенные данные concat_noise_x через промежуточную часть сети, представленную экземпляром класса IntermediateDnCNN, и получает промежуточное представление данных h_dncnn.
		pred_noise = self.upsamplefeatures(h_dncnn)			#Эта строка передает промежуточное представление данных h_dncnn через последний слой UpSampleFeatures, который выполняет восстановление размеров признаков до исходных размеров.
		return pred_noise		#Этот оператор возвращает предполагаемый уровень шума после обработки, представленный переменной pred_noise.
#Этот метод forward реализует процесс обработки входных данных через всю архитектуру модели FFDNet и возвращает результат обработки.