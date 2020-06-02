from imutils import face_utils
import cv2
import time
import numpy as np
import dlib
import wget
import os
from utils import extract_closest_face, eye_aspect_ratio
from collections import deque
class Sleeptor():
    def __init__(self, featurizer, modelo):
        self.featurizer = featurizer
        self.data_dir = 'data/'
        self.detector = dlib.get_frontal_face_detector()
        hog_filename = self.data_dir + 'shape_predictor_68_face_landmarks.dat'
        if(not os.path.isfile(hog_filename)):
            url = 'https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat?raw=true'
            wget.download(url, hog_filename)
        self.predictor = dlib.shape_predictor(hog_filename)

        self.modelo = modelo.model
        self.alerta_1 = cv2.imread('data/alertas/alerta.jpg')
        self.alerta_2 = cv2.imread('data/alertas/alerta_2.jpg')
        self.pestaneos = deque()
        self.ear_iniciales = []
        self.mean = None
        self.acum = 0

    def predict(self, data):
        result =  self.modelo.predict_classes(data)
        if result == 1:
            result_string = "Somnoliento"
        else:
            result_string = "Atento"

        return result_string

    def blinks(self, ls):

        earl = eye_aspect_ratio(ls[36:42])
        earr = eye_aspect_ratio(ls[42:48])
        ear = np.mean((earl, earr))
        if(len(self.ear_iniciales)<self.featurizer.size):
            self.ear_iniciales.append(ear)
            if(len(self.ear_iniciales)==self.featurizer.size):
                self.mean = np.mean(self.ear_iniciales)

        image = None
        t = time.time()
        if(self.mean is not None):
            if( ear < self.mean*0.7):
                if(self.acum == 0):
                    self.pestaneos.append(t)
                if(self.acum > 5):
                    image = self.alerta_1
                    print('\a')
                else:
                    self.acum += 1
            else:
                self.acum = 0

            n_pops = 0
            for pest in self.pestaneos:
                if t-pest > 10:
                   n_pops += 1
                else:
                    break

            for _ in range(n_pops):
                self.pestaneos.popleft()

                if(len(self.pestaneos) > 5):
                    image = self.alerta_2
                    print('\a')
        return image

    def live(self):
        # opciones de texto de resultado
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,400)
        upperLeftCorner        = (10,10)
        upperRightCorner       = (630, 10)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2

        prev = 0
        cap = cv2.VideoCapture(0)
        ventana = []
        result_string = 'Esperando completar ventana'

        while True:
            time_elapsed = time.time() - prev
            _, image = cap.read()

            if time_elapsed > 1./self.featurizer.frame_rate:
                prev = time.time()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = self.featurizer.detector(image, 0)
                if(len(rects)>0):
                    if(len(rects)==1):
                        face = rects[0]
                    else:
                        face = extract_closest_face(rects)
                    shape = self.predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)
                    alert = self.blinks(shape)
                    
                    gray = gray[max(face.top(),0):max(face.bottom(),0), max(face.left(),0):max(face.right(),0)]
                    gray = cv2.resize(gray, (self.featurizer.height, self.featurizer.width))
                    gray = (gray-128)/128
                    ventana.append(gray)

                    # si se cumple el step
                    if(len(ventana)==self.featurizer.size):
                        ventana_np = np.expand_dims(np.expand_dims(np.array(ventana),axis=0),axis=-1)
                        result_string = self.predict(ventana_np)
                        ventana = ventana[self.featurizer.step:]
                    tl = face.tl_corner()
                    br = face.br_corner()
                    cv2.rectangle(image, (tl.x,tl.y), (br.x,br.y), (0,255,0))
            if(alert is None):
                cv2.putText(image, result_string, bottomLeftCornerOfText, font, fontScale, fontColor,lineType)
            else:
                image = alert
            cv2.imshow("Monitor", image)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
        cap.release()
