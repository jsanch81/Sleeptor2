from imutils import face_utils
from sklearn.neighbors import KNeighborsClassifier
import dlib
import cv2
import numpy as np
import os
import pickle

def eye_aspect_ratio(eye):
    A = np.linalg.norm((eye[1] - eye[5]),2)
    B = np.linalg.norm((eye[2] - eye[4]),2)
    C = np.linalg.norm((eye[0] - eye[3]),2)
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm((mouth[14] - mouth[18]),2)
    C = np.linalg.norm((mouth[12] - mouth[16]),2)
    mar = (A ) / (C)
    return mar

def circularity(eye):
    A = np.linalg.norm((eye[1] - eye[4]),2)
    radius  = A/2.0
    Area = np.pi * (radius ** 2)
    p = 0
    p += np.linalg.norm((eye[0] - eye[1]),2)
    p += np.linalg.norm((eye[1] - eye[2]),2)
    p += np.linalg.norm((eye[2] - eye[3]),2)
    p += np.linalg.norm((eye[3] - eye[4]),2)
    p += np.linalg.norm((eye[4] - eye[5]),2)
    p += np.linalg.norm((eye[5] - eye[0]),2)
    return 4 * np.pi * Area /(p**2)



 class Sleeptor():

    def __init__(self):
        self.knn = pickle.load('data/models/knn.pkl');
        self.data = []
        self.n_features = 5


    # def calibration(self):
    #     cap = cv2.VideoCapture(0)
    #
    #     while True:
    #         # Getting out image by webcam
    #         _, image = cap.read()
    #         # Converting the image to gray scale
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         # Get faces into webcam's image
    #         rects = detector(gray, 0)
    #
    #         # For each detected face, find the landmark.
    #         for (i, rect) in enumerate(rects):
    #             # Make the prediction and transfom it to numpy array
    #             shape = predictor(gray, rect)
    #             shape = face_utils.shape_to_np(shape)
    #             data.append(shape)
    #             cv2.putText(image,"Calibrating...", bottomLeftCornerOfText, font, fontScale, fontColor,lineType)
    #
    #             # Draw on our image, all the finded cordinate points (x,y)
    #             for (x, y) in shape:
    #                 cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    #
    #         # Show the image
    #         cv2.imshow("Output", image)
    #
    #         k = cv2.waitKey(5) & 0xFF
    #         if k == 27:
    #             break
    #
    #     cv2.destroyAllWindows()
    #     cap.release()
    #
    #
    #     features_test = []
    #     for d in data:
    #         leye = d[36:42]
    #         reye = d[42:48]
    #         mouth = d[48:68]
    #         earl = eye_aspect_ratio(leye)
    #         earr = eye_aspect_ratio(reye)
    #         mar = mouth_aspect_ratio(eye)
    #         cir = (circularity(leye) + circularity(reye))/2
    #         mouth_eye = mar/ear
    #         features_test.append([earl, earr, mar, cir, mouth_eye])
    #
    #     features_test = np.array(features_test)
    #     x = features_test
    #     y = pd.DataFrame(x,columns=["EARL","EARR","MAR","Circularity","MOE"])
    #     df_means = y.mean(axis=0)
    #     df_std = y.std(axis=0)
    #
    #     return df_means,df_std

    def model(self, data):
        Result =  self.knn.predict(data)
        if Result == 1:
            Result_String = "Drowsy"
        else:
            Result_String = "Alert"

        return Result_String

    def featurize(self, landmarks):
        features = np.empty((0, self.n_features))
        for d in landmarks:
            leye = d[36:42]
            reye = d[42:48]
            mouth = d[48:68]
            ear = (eye_aspect_ratio(leye) + eye_aspect_ratio(reye))/2
            earl = eye_aspect_ratio(leye)
            earr = eye_aspect_ratio(reye)
            mar = mouth_aspect_ratio(mouth)
            cir = (circularity(leye) + circularity(reye))/2
            mouth_eye = mar/ear
            features = np.append(features, [[earl, earr, mar, cir, mouth_eye]], axis=0)
        return features

    def live(self):
        frame_rate = 10
        prev = 0
        cap = cv2.VideoCapture(0)
        ventana = []
        count = 0;
        while True:
            time_elapsed = time.time() - prev
            _, image = cap.read()
            if time_elapsed > 1./frame_rate:
                prev = time.time()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = detector(image, 0)
                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    features = self.featurize(shape);
                    if (count < 10):
                        ventana.append(features)
                    else:
                        ventana_np = np.array(ventana)
                        data = ventana_np.ravel()
                        Result_String = self.model(data)
                        cv2.putText(image,Result_String, bottomLeftCornerOfText, font, fontScale, fontColor,lineType)
                        count = 0
                        ventana = []
                    count +=1

                    for (x, y) in shape:
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            cv2.imshow("Output", image)

            k = cv2.waitKey(300) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
        cap.release()
        return data,result
