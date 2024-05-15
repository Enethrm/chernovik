import numpy as np #Numpy используется для работы с многомерными массивами и математическими функциями
import cv2          # предоставляет множество функций для компьютерного зрения и обработки изображений.
import os           # предоставляет функции для работы с операционной системой, в данном случае, для работы с файлами и путями

from fastapi import APIRouter, UploadFile # Импорт классов APIRouter и UploadFile из библиотеки FastAPI. FastAPI используется для создания веб-сервисов и API.
from fastapi.responses import FileResponse # Импорт класса FileResponse из fastapi.responses. Он используется для отправки файлов в ответ на запросы.


colorization_router = APIRouter(prefix='/colorization') # Создание экземпляра APIRouter с префиксом /colorization. Это определяет префикс для всех маршрутов, определенных внутри этого роутера.


@colorization_router.post('/') # Декоратор, который привязывает функцию к определенному HTTP-методу (POST) и URL-адресу (/) в рамках маршрутизатора colorization_router.
async def colorization_proccess(file: UploadFile): # Определение асинхронной функции colorization_process, которая принимает файл в качестве параметра.
    path_to_file = f'./temp/{file.filename}'        # Форматированная строка, которая создает путь к файлу на основе имени файла, загруженного пользователем, в папке temp.
    with open(path_to_file, 'wb+') as file_obj:     # Открытие файла для записи в бинарном режиме.
        file_obj.write(file.file.read())            # Запись содержимого загруженного файла в созданный файл.


    PROTOTXT = './source/colorization_model/colorization_deploy_v2.prototxt'
    POINTS = './source/colorization_model/pts_in_hull.npy'
    MODEL = './source/colorization_model/colorization_release_v2.caffemodel'


    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    class8 = net.getLayerId("class8_ab")    #Получение идентификатора слоя с именем "class8_ab" из объекта net. Этот слой отвечает за предсказание AB-каналов.
    conv8 = net.getLayerId("conv8_313_rh")  #Получение идентификатора слоя с именем "conv8_313_rh" из объекта net. Этот слой отвечает за предсказание вероятностей для 313 точек в пространстве цветов.
    pts = pts.transpose().reshape(2, 313, 1, 1) #Транспонирование массива pts и изменение его формы до (2, 313, 1, 1). Этот массив содержит координаты цветовых точек в пространстве AB.
    net.getLayer(class8).blobs = [pts.astype("float32")]    #Установка атрибута blobs для слоя с идентификатором class8. В этот атрибут передается массив pts в формате float32, который содержит координаты цветовых точек.
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")] #Установка атрибута blobs для слоя с идентификатором conv8. В этот атрибут передается массив, заполненный значением 2.606, который представляет собой начальные значения для предсказанных вероятностей в пространстве цветов.

    image = cv2.imread(path_to_file)            # Преобразование изображения в формат LAB и его подготовка к подаче на вход модели.
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224)) 
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))      # Получение предсказанных AB-каналов из модели для входного изображения.
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0])) 

    L = cv2.split(lab)[0]                       # Объединение предсказанных AB-каналов с L-каналом в формате LAB.
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2) 

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR) # Преобразование изображения из формата LAB обратно в формат BGR.
    colorized = np.clip(colorized, 0, 1)    # Корректировка значений пикселей изображения и их ограничение в диапазоне от 0 до 1.

    colorized = (255 * colorized).astype("uint8")   # Преобразование значений пикселей изображения к типу uint8.

    cv2.imwrite(path_to_file, colorized) #Сохранение цветизированного изображения в файл.

    return FileResponse(path=path_to_file) # Возврат цветизированного файла в качестве ответа на запрос.