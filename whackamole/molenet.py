from keras.models import model_from_json
import cv2 as cv
import numpy as np

mole_net = model_from_json(open("molenet/model.json").read())
mole_net.load_weights("molenet/weights.hdf5")


def change_shape(image):
    tmp = cv.resize(image, (600, 400))
    ret = np.zeros((1, 400, 600, 3), dtype=np.float32)
    ret[0] = tmp/255
    return ret


def mole_net_predict(image):
    input_image = change_shape(image)
    p = mole_net.predict(input_image)
    prediction = np.uint8(p[0] * 255)
    _, prediction = cv.threshold(prediction, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return prediction

