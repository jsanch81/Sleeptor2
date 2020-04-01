import pandas as pd
from imutils import face_utils
import dlib
import cv2
import numpy as np
import time
import os
import wget
from collections import deque
#from pydub import AudioSegment

detector = dlib.get_frontal_face_detector()
hog_filename = "data/shape_predictor_68_face_landmarks.dat"
if(not os.path.isfile(hog_filename)):
    url = 'https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat?raw=true'
    wget.download(url, hog_filename)
predictor = dlib.shape_predictor(hog_filename)

def eye_aspect_ratio(eye):
	A = np.linalg.norm((eye[1], eye[5]),2)
	B = np.linalg.norm((eye[2], eye[4]),2)
	C = np.linalg.norm((eye[0], eye[3]),2)
	ear = (A + B) / (2.0 * C)
	return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm((mouth[14], mouth[18]),2)
    C = np.linalg.norm((mouth[12], mouth[16]),2)
    mar = (A ) / (C)
    return mar

def circularity(eye):
    A = np.linalg.norm((eye[1], eye[4]),2)
    radius  = A/2.0
    Area = np.pi * (radius ** 2)
    p = 0
    p += np.linalg.norm((eye[0], eye[1]),2)
    p += np.linalg.norm((eye[1], eye[2]),2)
    p += np.linalg.norm((eye[2], eye[3]),2)
    p += np.linalg.norm((eye[3], eye[4]),2)
    p += np.linalg.norm((eye[4], eye[5]),2)
    p += np.linalg.norm((eye[5], eye[0]),2)
    return 4 * np.pi * Area /(p**2)

def mouth_over_eye(eye):
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    mouth_eye = mar/ear
    return mouth_eye



def getFrame(sec,vidcap):
    start = 180000
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start + sec*1000)
    hasFrames,image = vidcap.read()
    return hasFrames, image


data = []
labels = []
for j in [10]:
  for i in [0,1]:
    vidcap = cv2.VideoCapture('dataset/' + str(j) +'/' + str(i) + '.mp4')
    sec = 0
    frameRate = 1
    success, image  = getFrame(sec,vidcap)
    count = 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    while success and count < 360:
          rects = detector(gray,0)
          if len(rects) == 1:
              count += 1
              shape = predictor(gray, rects[0])
              shape = face_utils.shape_to_np(shape)
              data.append(shape)
              labels.append([int(i)])
              sec = sec + frameRate
              sec = round(sec, 2)
              success, image = getFrame(sec,vidcap)
              gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
              print(count)
          else:
              sec = sec + frameRate
              sec = round(sec, 2)
              success, image = getFrame(sec,vidcap)
              gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
              print("not detected")

data = np.array(data)
labels = np.array(labels)

features = []
for d in data:
  eye = d[36:68]
  ear = eye_aspect_ratio(eye)
  mar = mouth_aspect_ratio(eye)
  cir = circularity(eye)
  mouth_eye = mouth_over_eye(eye)
  features.append([ear, mar, cir, mouth_eye])

features = np.array(features)
features.shape

np.save(open('Data_60_10.npy', 'wb'),data)
np.save(open('Fold5_part2_features_60_10.npy', 'wb'),features)
np.save(open('Fold5_part2_labels_60_10.npy', 'wb'),labels)
df_label = pd.DataFrame(labels,columns=['Y'])
df_feature = pd.DataFrame(features, columns=["EAR","MAR","Circularity","MOE"])
df_total = pd.concat([df_feature,df_label], axis=1)
print(df_total.shape)
df_total.to_csv('feature_labels.csv')
df_label.to_csv('Fold5_part2_labels_60_10.csv')
df_feature.to_csv('Fold5_part2_features_60_10.csv')
#np.savetxt("Fold5_part2_features_60_10.csv", features, delimiter = ",")
#np.savetxt("Fold5_part2_labels_60_10.csv", labels, delimiter = ",")
