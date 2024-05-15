# В функции super_resolution_handler выполняются следующие шаги:

# Сохранение загруженного файла во временный каталог.
# Загрузка предварительно обученной нейросетевой модели для увеличения разрешения изображений.
# Предобработка изображения, включая чтение изображения с помощью OpenCV, преобразование его в формат torch.Tensor, нормализацию и отправку на устройство (GPU или CPU).
# Применение модели для увеличения разрешения изображения.
# Постобработка результата, включая возврат изображения в формат OpenCV, сохранение его на диск и отправку клиенту в качестве ответа.
# Этот скрипт предназначен для использования в среде API, где он будет обрабатывать загружаемые изображения, применяя к ним метод увеличения разрешения с помощью нейросетевой модели.


import os.path #Для работы с путями к файлам и директориями.
import torch #Библиотека PyTorch для глубокого обучения.

from .utils_image import utils_image as util

import cv2
import numpy as np
from .network_rrdbnet import RRDBNet as net
from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse




super_resolution_router = APIRouter(prefix='/super_resolution')
@super_resolution_router.post('/')
async def super_resolution_handler(file: UploadFile):
    path_to_file = f'./temp/{file.filename}'
    with open(path_to_file, 'wb+') as file_obj:
        file_obj.write(file.file.read())

    model_path = './source/super_resolution_model/BSRGAN.pth'  # set model path | 'BSRGANx2' for scale factor 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Устройство, на котором будет выполняться модель (GPU или CPU).

    torch.cuda.empty_cache()

    # define network and load model
    sf = 4              # Масштабный коэффициент для преобразования разрешения изображения.
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()    # Модель помечается как оценочная (evaluation) и перемещается на устройство (GPU или CPU).
    model = model.to(device)
    torch.cuda.empty_cache()

    img = cv2.imread(path_to_file, cv2.IMREAD_COLOR) #Исходное изображение загружается с диска с помощью OpenCV. Флаг cv2.IMREAD_COLOR указывает на то, что изображение будет загружено в цветовом формате.

    img = img * 1.0 / 255 # Предобрабатывается изображение путем нормализации значений пикселей. Это делается путем умножения каждого значения пикселя на 1.0 и деления на 255, чтобы значения пикселей оказались в диапазоне от 0 до 1.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float() #Изображение конвертируется в формат torch.Tensor. Сначала происходит транспонирование каналов изображения с помощью np.transpose, чтобы изменить порядок каналов из BGR (Blue, Green, Red) на RGB (Red, Green, Blue). Затем массив NumPy преобразуется в тензор PyTorch с помощью torch.from_numpy, и тип данных тензора приводится к float.
    img_L = img.unsqueeze(0) # Создается новый тензор, добавляя размерность батча в начало. Это делается с помощью метода unsqueeze(0), чтобы получить тензор формы [1, C, H, W], где C - количество каналов, H - высота изображения и W - ширина изображения.
    img_L = img_L.to(device) # Тензор перемещается на указанное устройство (GPU или CPU), указанное в переменной device.

    with torch.no_grad(): # Запускается блок кода, в котором вычисляется выход модели. Оператор with torch.no_grad() указывает на то, что градиенты не будут вычисляться для этого блока, что ускоряет процесс и экономит память.
        output = model(img_L).data.squeeze().float().cpu().clamp_(0, 1).numpy() #Используется нейросетевая модель для получения выходных данных. Результат преобразуется в формат NumPy.
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) # Выполняется обратное преобразование каналов изображения из RGB в BGR, чтобы сохранить его в формате, который принимает функция cv2.imwrite.
    output = (output * 255.0).round() # Происходит обратная нормализация, чтобы значения пикселей находились в диапазоне от 0 до 255, и округление значений до целых чисел.
    cv2.imwrite(path_to_file, output) # Обработанное изображение сохраняется на диск с использованием OpenCV.
    return FileResponse(path=path_to_file)
