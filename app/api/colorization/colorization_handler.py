import numpy as np
import cv2
import os

from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse


colorization_router = APIRouter(prefix='/colorization')


@colorization_router.post('/')
async def colorization_proccess(file: UploadFile):
    path_to_file = f'./temp/{file.filename}' 
    with open(path_to_file, 'wb+') as file_obj:
        file_obj.write(file.file.read())


    PROTOTXT = './source/colorization_model/colorization_deploy_v2.prototxt'
    POINTS = './source/colorization_model/pts_in_hull.npy'
    MODEL = './source/colorization_model/colorization_release_v2.caffemodel'


    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    image = cv2.imread(path_to_file)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    # сохранить numpy array как Image
    cv2.imwrite(path_to_file, colorized)

    

    return FileResponse(path=path_to_file)