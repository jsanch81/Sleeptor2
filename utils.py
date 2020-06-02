import numpy as np
import cv2
import math
from scipy.spatial.distance import cdist, euclidean

def get_frame(sec, cap):
    start = 0
    cap.set(cv2.CAP_PROP_POS_MSEC, start + sec*1000)
    hasFrames, image = cap.read()
    limit = 640
    if(hasFrames):
        # mirar si alguna dimension es mayor al limite definido
        if(image.shape[0] > limit or image.shape[1] > limit):
            # encontrar la mayor dimension y reducirla al limite maximo definido.
            if(image.shape[0] > image.shape[1]):
                scaling_factor = image.shape[0]/limit
            else:
                scaling_factor = image.shape[1]/limit
            # cambiar tamaño a la imagen de acuerdo al limtie definido
            image = cv2.resize(image, (int(image.shape[1]/scaling_factor), int(image.shape[0]/scaling_factor)), interpolation = cv2.INTER_AREA)
    return hasFrames, image

def extract_closest_face(faces):
    areas = []
    for f in faces:
        areas.append(f.height() * f.width())

    idx = np.argmax(areas)
    return faces[idx]

def normalize(data):
    means = np.nanmean(data, axis=1).reshape(-1,1)
    stds = np.nanstd(data, axis=1).reshape(-1,1)
    data_s = (data-means)/stds
    return data_s
    
def eye_aspect_ratio(eye):
    A = np.linalg.norm((eye[1] - eye[5]),2)
    B = np.linalg.norm((eye[2] - eye[4]),2)
    C = np.linalg.norm((eye[0] - eye[3]),2)
    ear = (A + B) / (2.0 * C)
    return ear
