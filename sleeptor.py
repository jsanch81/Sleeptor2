from imutils import face_utils
import cv2
import time
import numpy as np
from utils import extract_closest_face

class Sleeptor():
    def __init__(self, featurizer, modelo):
        self.featurizer = featurizer
        self.modelo = modelo.model
        self.alerta_1 = cv2.imread('data/alertas/alerta.jpg')
        self.alerta_2 = cv2.imread('data/alertas/alerta_2.jpg')

    def predict(self, data):
        result =  self.modelo.predict_classes(data)
        if result == 1:
            result_string = "Somnoliento"
        else:
            result_string = "Atento"

        return result_string

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

            cv2.putText(image,result_string, bottomLeftCornerOfText, font, fontScale, fontColor,lineType)
            cv2.imshow("Monitor", image)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
        cap.release()
