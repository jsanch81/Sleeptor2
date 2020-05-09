from imutils import face_utils
import cv2
import time
import numpy as np

class Sleeptor():
    def __init__(self, featurizer, modelo):
        self.featurizer = featurizer
        self.modelo = modelo.model
        self.alerta_1 = cv2.imread('data/alertas/alerta.jpg')
        self.alerta_2 = cv2.imread('data/alertas/alerta_2.jpg')

    def predict(self, data):
        result =  self.modelo.predict(data)
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

        while True:
            time_elapsed = time.time() - prev
            _, image = cap.read()
            if time_elapsed > 1./self.featurizer.frame_rate:
                prev = time.time()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = self.featurizer.detector(image, 0)
                for (i, rect) in enumerate(rects):
                    shape = self.featurizer.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    features, xl, xr = self.featurizer.featurize([shape], [gray])
                    ventana.append(features)
                    if (len(ventana) == self.featurizer.size):
                        ventana_np = np.array(ventana)
                        data = ventana_np.reshape(1, -1)
                        result_string = self.predict(data)
                        cv2.putText(image,result_string, bottomLeftCornerOfText, font, fontScale, fontColor,lineType)
                        if(xl and (xl < 0.3 or xl > 0.7)):
                            image = self.alerta_1
                        if(xr and (xr < 0.3 or xr > 0.7)):
                            image = self.alerta_1
                        ventana = ventana[self.featurizer.step:]

                    for (x, y) in shape:
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            cv2.imshow("Monitor", image)

            k = cv2.waitKey(300) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
        cap.release()
