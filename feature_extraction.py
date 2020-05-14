from imutils import face_utils
import dlib
import cv2
import numpy as np
import os
import wget
import multiprocessing as mp
from functools import partial
from utils import eye_aspect_ratio, mouth_aspect_ratio, circularity, calculate_threshold
from utils import get_frame, calculate_eye_coords, get_useful_triangles, get_cosines, extract_closest_face

class Featurizer():
    def __init__(self):
        self.data_dir = 'data/'
        self.detector = dlib.get_frontal_face_detector()
        hog_filename = self.data_dir + 'shape_predictor_68_face_landmarks.dat'
        if(not os.path.isfile(hog_filename)):
            url = 'https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat?raw=true'
            wget.download(url, hog_filename)
        self.predictor = dlib.shape_predictor(hog_filename)
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 1500 # Because no pupil has area bigger than 1500 pixels
        self.blob_detector = cv2.SimpleBlobDetector_create(detector_params)
        self.n_landmarks = 68
        self.n_features = 5 #+ 54 # 3 cosines for each one of the 18 relevant triangles in face = 54
        # cuantos frames de cada video se van a capturar
        self.n_samples_per_video = 3000
        # cuantos frames por segundo se van a capturar
        self.frame_rate = 5
        # tamaño de la ventana deslizante
        self.size = 100
        # paso de la ventana deslizante
        self.step = 1
        assert self.n_samples_per_video > self.size + self.step

    def calculate_rotation(self, gray):
        rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        rects = self.detector(gray,0)
        cont = 0
        while len(rects) < 1 and cont < len(rotations)-1:
            cont += 1
            gray = cv2.rotate(gray, rotations[cont])
            rects = self.detector(gray,0)
        return rotations[cont]

    def process_eye(self, img, coords, threshold):
        """
        :param img: gray image frame
        :param coords: coords of the eye in the frame
        :param threshold: threshold value for threshold function

        :return: keypoints
        """
        x,y,w,h = coords
        img = img[y:y+h, x:x+w]
        #cv2.imwrite('tmp.jpg', img)
        _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        #cv2.imwrite('thresh.jpg', img)
        img = cv2.dilate(img, None, iterations=2)
        img = cv2.erode(img, None, iterations=2)
        #cv2.imwrite('erod_dila.jpg', img)
        img = cv2.medianBlur(img, 7)
        #cv2.imwrite('proc.jpg', img)
        
        keypoints = self.blob_detector.detect(img)

        return keypoints

    def extract(self, fold, person):
        data_filename = self.data_dir + fold + '/' + person + '/data_' + str(self.n_samples_per_video) + '.npy'
        if(not os.path.isfile(data_filename)):
            landmarks = []
            labels = []
            grays = []
            for i in [0,1]:
                cap = cv2.VideoCapture(self.data_dir + fold + '/' + person + '/' + str(i) + '.mp4')
                sec = 0
                step = 1.0/self.frame_rate
                success, image = get_frame(sec, cap)
                count = 0
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rotation = self.calculate_rotation(gray)
                count_bads = 0
                bad = False
                while success and count < self.n_samples_per_video:
                    if(rotation is not None):
                        gray = cv2.rotate(gray, rotation)
                    grays.append(gray)
                    rects = self.detector(gray,0)
                    if(len(rects) >= 1):
                        if(len(rects)==1):
                            face = rects[0]
                        else:
                            face = extract_closest_face(rects)
                        count += 1
                        shape = self.predictor(gray, face)
                        shape = face_utils.shape_to_np(shape)
                        landmarks.append(shape)
                        labels.append([float(i)])
                        sec = sec + step
                        sec = round(sec, 2)
                        success, image = get_frame(sec, cap)
                        if(success):
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        count_bads += 1
                        if(count_bads > self.n_samples_per_video/50):
                            bad = True
                            break
                        print("face not detected for fold: %s, person: %s, video:%d.mp4, at second: %.1f" % (fold, person, i, sec))
                        sec = sec + step
                        sec = round(sec, 2)
                        success, image = get_frame(sec, cap)
                        if(success):
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if(bad):
                    break

            if(count == self.n_samples_per_video):
                landmarks = np.array(landmarks)
                labels = np.array(labels)
                data = np.append(landmarks, np.append(labels.reshape(-1, 1, 1), np.zeros((labels.shape[0], 1, 1)), axis=2), axis=1)
                np.save(data_filename, data)
            else:
                data = np.ones((1,1))
                np.save(data_filename, data)
                landmarks = None
                labels = None
                grays = None
        else:
            data = np.load(data_filename, allow_pickle=True)
            if(len(data)==self.n_samples_per_video*2):
                assert data.shape[1] == self.n_landmarks + 1
                landmarks = data[:,:self.n_landmarks,:]
                labels = data[:,[-1], 0]
            else:
                landmarks = None
                labels = None
            grays = None
        return landmarks, labels, grays

    def featurize(self, landmarks, grays):
        features = np.empty((0, self.n_features))
        objects = zip(landmarks, grays) if grays is not None and len(grays) > 0 else zip(landmarks, [0 for _ in range(len(landmarks))])
        for ls_i, gray in objects:
            leye = ls_i[36:42]
            reye = ls_i[42:48]
            mouth = ls_i[48:68]
            ear = (eye_aspect_ratio(leye) + eye_aspect_ratio(reye))/2
            earl = eye_aspect_ratio(leye)
            earr = eye_aspect_ratio(reye)
            mar = mouth_aspect_ratio(mouth)
            cir = (circularity(leye) + circularity(reye))/2
            mouth_eye = mar/ear

            xl = None
            xr = None

            #triangles = get_useful_triangles(ls_i)
            #cosines = []
            #for triangle in triangles:
            #    tmp = get_cosines(triangle)
            #    cosines.extend(tmp)

            if(grays is not None):
                # detectar posicion de la pupila
                right_eye_coords = calculate_eye_coords(ls_i, 36,37,38,39,40,41)
                left_eye_coords = calculate_eye_coords(ls_i, 42,43,44,45,46,47)

                # distancias de la apertura de los ojos
                d1 = np.linalg.norm((ls_i[37] - ls_i[41]), 2)
                d2 = np.linalg.norm((ls_i[38] - ls_i[40]), 2)
                d3 = np.linalg.norm((ls_i[43] - ls_i[47]), 2)
                d4 = np.linalg.norm((ls_i[44] - ls_i[46]), 2)

                # umbral para definir color blanco de los ojos
                threshold = calculate_threshold(gray, ls_i, d1, d2, d3, d4)
        
                # ubicar pupila izquierda
                keypoints = self.process_eye(gray, left_eye_coords, threshold)
                if(len(keypoints) > 0):
                    xl = keypoints[0].pt[0]/left_eye_coords[2]
                    yl = keypoints[0].pt[1]/left_eye_coords[3]
                else:
                    xl = None
                    yl = None
                
                # ubicar pupila derecha
                keypoints = self.process_eye(gray, right_eye_coords, threshold)
                if(len(keypoints) > 0):
                    xr = keypoints[0].pt[0]/right_eye_coords[2]
                    yr = keypoints[0].pt[1]/right_eye_coords[3]
                else:
                    xr = None
                    yr = None

            # features = np.append(features, [[earl, earr, mar, cir, mouth_eye].extend(cosines)], axis=0)
            features = np.append(features, [[earl, earr, mar, cir, mouth_eye]], axis=0)
        return features, xl, xr

    def run(self):
        data_filename = self.data_dir + 'data_' + str(self.size) + '_' + str(self.step) + '_' + str(self.n_samples_per_video) + '.npy'
        medias_filename = self.data_dir + 'medias_' + str(self.size) + '_' + str(self.step) + '_' + str(self.n_samples_per_video) + '.npy'
        if(not os.path.isfile(data_filename)):
            # eliminar por que no permite paralelizar este objeto
            del self.blob_detector

            data = np.empty((0, self.n_features*self.size+1), dtype=np.float32)
            medias = np.empty((0, self.n_features+1), dtype=np.float32)
            folds = [x for x in os.listdir('data/') if x.startswith('Fold') and not '.zip' in x]

            for fold in folds:
                people = [x for x in os.listdir('data/'+fold) if x.isnumeric()]
                part = partial(self.extract_person, fold)

                pool = mp.Pool(min(6, mp.cpu_count()-3))
                results = list(pool.imap(part, [person for person in people]))
                pool.close()
                    
                
                for data_p, medias_p in results:
                    if(data_p is not None):
                        data = np.append(data, data_p, axis=0)
                        medias = np.append(medias, medias_p, axis=0)

            np.save(data_filename, data)
            np.save(medias_filename, medias)

            # volver a crear el objeto elimiando previamente
            detector_params = cv2.SimpleBlobDetector_Params()
            detector_params.filterByArea = True
            detector_params.maxArea = 1500 # Because no pupil has area bigger than 1500 pixels
            self.blob_detector = cv2.SimpleBlobDetector_create(detector_params)

        else:
            data = np.load(data_filename, allow_pickle=True)
            medias = np.load(medias_filename, allow_pickle=True)
            assert data.shape[1] == self.n_features*self.size + 1

        return data, medias

    def extract_person(self, fold, person):
        landmarks_p, labels_p, grays = self.extract(fold, person)
        if(landmarks_p is not None):
            features_p, _, _ = self.featurize(landmarks_p, None)
            part = list(labels_p).index(1)
            atento = features_p[:part]
            vents_atento, meds_atento = self.serializer(atento, 0)
            dormido = features_p[part:]
            vents_dormido, meds_dormido = self.serializer(dormido, 1)
            data_p = np.append(vents_dormido, vents_atento, axis=0).squeeze()
            medias_p = np.append(meds_atento, meds_dormido, axis=0).squeeze()
        else:
            data_p = None
            medias_p = None
        
        return [data_p, medias_p]

    def serializer(self, X, y):

        ventanas = []
        medias = []

        for i in range(0, len(X)-self.size, self.step):
            ventana = X[i:i+self.size, :]
            medias_i = np.mean(ventana, axis=0)
            ventanas.append(np.concatenate((ventana.reshape(1, -1), [[y]]), axis=1))
            medias.append(np.concatenate((medias_i.reshape(1, -1), [[y]]), axis=1))

        ventanas = np.array(ventanas)
        medias = np.array(medias)

        return ventanas, medias
