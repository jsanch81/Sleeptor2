from imutils import face_utils
import dlib
import cv2
import numpy as np
import os
import wget

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

def getFrame(sec, cap):
    start = 0
    cap.set(cv2.CAP_PROP_POS_MSEC, start + sec*1000)
    hasFrames, image = cap.read()
    return hasFrames, image

class Featurizer():
    def __init__(self):
        self.data_dir = 'data/'
        self.detector = dlib.get_frontal_face_detector()
        hog_filename = self.data_dir + 'shape_predictor_68_face_landmarks.dat'
        if(not os.path.isfile(hog_filename)):
            url = 'https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat?raw=true'
            wget.download(url, hog_filename)
        self.predictor = dlib.shape_predictor(hog_filename)
        self.n_landmarks = 68
        self.n_features = 5

    def calculate_rotation(self, gray):
        rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        rects = self.detector(gray,0)
        cont = 0
        while len(rects) < 1 and cont < len(rotations):
            cont += 1
            gray = cv2.rotate(gray, rotations[cont])
            rects = self.detector(gray,0)
        return rotations[cont]

    def extract(self, fold, person):
        if(not os.path.isfile(self.data_dir + fold + '/' + person + '/data.npy')):
            landmarks = []
            labels = []
            for i in [0,1]:
                cap = cv2.VideoCapture(self.data_dir + fold + '/' + person + '/' + str(i) + '.mp4')
                sec = 0
                frameRate = 0.1
                success, image = getFrame(sec, cap)
                count = 0
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rotation = self.calculate_rotation(gray)
                while success and count < 4000:
                    if(rotation is not None):
                        gray = cv2.rotate(gray, rotation)
                    rects = self.detector(gray,0)
                    if len(rects) == 1:
                        count += 1
                        shape = self.predictor(gray, rects[0])
                        shape = face_utils.shape_to_np(shape)
                        landmarks.append(shape)
                        labels.append([float(i)])
                        sec = sec + frameRate
                        sec = round(sec, 2)
                        success, image = getFrame(sec, cap)
                        if(success):
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        print("face not detected for fold: %s, person: %s, video:%d.mp4, at second: %.0f" % (fold, person, i, sec))
                        sec = sec + frameRate
                        sec = round(sec, 2)
                        success, image = getFrame(sec, cap)
                        if(success):
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            landmarks = np.array(landmarks)
            labels = np.array(labels)
            data = np.append(landmarks, np.append(labels.reshape(-1, 1, 1), np.zeros((labels.shape[0], 1, 1)), axis=2), axis=1)
            np.save(self.data_dir + fold + '/' + person + '/data.npy', data)
        else:
            data = np.load(self.data_dir + fold + '/' + person + '/data.npy', allow_pickle=True)
            assert data.shape[1] == self.n_landmarks + 1
            landmarks = data[:,:self.n_landmarks,:]
            labels = data[:,[-1], 0]
        return landmarks, labels

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

    def run(self):
        size = 10
        step = 1
        if(not os.path.isfile(self.data_dir + 'data.npy')):
            data = np.empty((0, self.n_features*size+1), dtype=np.float32)
            medias = np.empty((0, self.n_features+1), dtype=np.float32)
            folds = [x for x in os.listdir('data/') if x.startswith('Fold')]

            for fold in folds:
                people = [x for x in os.listdir('data/'+fold) if x.isnumeric()]
                for person in people:
                    landmarks_p, labels_p = self.extract(fold, person)
                    features_p = self.featurize(landmarks_p)
                    part = list(labels_p).index(1)
                    atento = features_p[:part]
                    vents_atento, meds_atento = self.serializer(atento, 0, step, size)
                    dormido = features_p[part:]
                    vents_dormido, meds_dormido = self.serializer(dormido, 1, step, size)
                    data_p = np.append(vents_dormido, vents_atento, axis=0).squeeze()
                    medias_p = np.append(meds_atento, meds_dormido, axis=0).squeeze()
                    data = np.append(data, data_p, axis=0)
                    medias = np.append(medias, medias_p, axis=0)
            
            np.save(self.data_dir + 'data.npy', data)
            np.save(self.data_dir + 'medias.npy', data)

        else:
            data = np.load(self.data_dir + 'data.npy', allow_pickle=True)
            medias = np.load(self.data_dir + 'medias.npy', allow_pickle=True)
            assert data.shape[1] == self.n_features*size + 1

        return data, medias
    
    def serializer(self, X, y, step, size):

        ventanas = []
        medias = []

        for i in range(0, len(X)-size, step):
            ventana = X[i:i+size, :]
            medias_i = np.mean(ventana, axis=0)
            ventanas.append(np.concatenate((ventana.reshape(1, -1), [[y]]), axis=1))
            medias.append(np.concatenate((medias_i.reshape(1, -1), [[y]]), axis=1))

        ventanas = np.array(ventanas)
        medias = np.array(medias)

        return ventanas, medias