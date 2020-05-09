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

def calculate_threshold(gray, ls, d1, d2, d3, d4):
        """
        Function that calculate the color of the white part of the eye. It calculates the minimum between the left and right white part of each eye.
        :param gray: gray image
        :param ls: list of landmarks
        Parameters d1 to d4 are the distance between the upper and the lower landmarks of the eye.
        :return threshold: threshold for binarizing image.
        """
        
        x1 = int((ls[36][0] + (ls[37][0] + ls[41][0])/2)/2)
        y1 = ls[36][1]
        t1 = gray[y1,x1]
        
        x2 = int((ls[39][0] + (ls[38][0] + ls[40][0])/2)/2)
        y2 = ls[39][1]
        t2 = gray[y2, x2]
        
        # takes the lowest if the difference is no longer that 10% else takes the highest.
        p1 = min(t1,t2) if (min(t1,t2) >= max(t1,t2)*0.9) else max(t1,t2)

        x3 = int((ls[42][0] + (ls[43][0] + ls[47][0])/2)/2)
        y3 = ls[42][1]
        t3 = gray[y3,x3]
        
        x4 = int((ls[45][0] + (ls[44][0] + ls[46][0])/2)/2)
        y4 = ls[45][1]
        t4 = gray[y4, x4]

        # takes the lowest if the difference is no longer that 10% else takes the highest.
        p2 = min(t3,t4) if (min(t3,t4) >= max(t3,t4)*0.9) else max(t3,t4)

        return int(min(p1,p2)*0.8)

def getFrame(sec, cap):
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


def calculate_eye_coords(ls, a,b,c,d,e,f):
    """
    Function that calculates the rectangle where the eye is located.
    :param ls: list of landmarks
    Params a to f are the number of the landmark of each eye, a is the left corner of the eye and they are clockwise ordered. 
    Then b is the upper left landmark, c the upper right and so on. d is the right corner of the eye.
    :return [x,y,w,h]
    """
    w = ls[d][0]-ls[a][0]
    h = (ls[e][1]+ls[f][1])/2.0 - (ls[b][1]+ls[c][1])/2.0
    x = ls[a][0] - w*0.4
    y = (ls[b][1]+ls[c][1])/2.0 - h
    w *= 1.8
    h *= 3.0
    return [int(x),int(y),int(w),int(h)]

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
        if(not os.path.isfile(self.data_dir + fold + '/' + person + '/data.npy')):
            landmarks = []
            labels = []
            grays = []
            for i in [0,1]:
                cap = cv2.VideoCapture(self.data_dir + fold + '/' + person + '/' + str(i) + '.mp4')
                sec = 0
                frameRate = 0.1
                success, image = getFrame(sec, cap)
                count = 0
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rotation = self.calculate_rotation(gray)
                while success and count < 3000:
                    if(rotation is not None):
                        gray = cv2.rotate(gray, rotation)
                    grays.append(gray)
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
                        print("face not detected for fold: %s, person: %s, video:%d.mp4, at second: %.1f" % (fold, person, i, sec))
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
        return landmarks, labels, grays

    def featurize(self, landmarks, grays):
        features = np.empty((0, self.n_features))
        for ls_i, gray in zip(landmarks, grays):
            leye = ls_i[36:42]
            reye = ls_i[42:48]
            mouth = ls_i[48:68]
            ear = (eye_aspect_ratio(leye) + eye_aspect_ratio(reye))/2
            earl = eye_aspect_ratio(leye)
            earr = eye_aspect_ratio(reye)
            mar = mouth_aspect_ratio(mouth)
            cir = (circularity(leye) + circularity(reye))/2
            mouth_eye = mar/ear

            # detectar posicion de la pupila
            right_eye_coords = calculate_eye_coords(ls_i, 36,37,38,39,40,41)
            left_eye_coords = calculate_eye_coords(ls_i, 42,43,44,45,46,47)

            # distancias de la apertura de los ojos
            d1 = np.linalg.norm((ls_i[37] - ls_i[41]), 2)
            d2 = np.linalg.norm((ls_i[38] - ls_i[40]), 2)
            d3 = np.linalg.norm((ls_i[43] - ls_i[47]), 2)
            d4 = np.linalg.norm((ls_i[44] - ls_i[46]), 2)

            # umbral apra definir color blanco de los ojos
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

            features = np.append(features, [[earl, earr, mar, cir, mouth_eye]], axis=0)
        return features, xl, xr

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
                    landmarks_p, labels_p, grays = self.extract(fold, person)
                    features_p, _, _ = self.featurize(landmarks_p, grays)
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
            np.save(self.data_dir + 'medias.npy', medias)

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
