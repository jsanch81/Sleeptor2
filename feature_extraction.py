import dlib
import cv2
import numpy as np
import os
import wget
from utils import get_frame, extract_closest_face

class Featurizer():
    def __init__(self):
        self.data_dir = 'data/'
        self.detector = dlib.get_frontal_face_detector()
        # cuantos frames de cada video se van a capturar
        self.n_samples_per_video = 200
        # cuantos frames por segundo se van a capturar
        self.frame_rate = 5
        # tamaño de la ventana deslizante
        self.size = 100
        # paso de la ventana deslizante
        self.step = 5

        self.height = 64
        self.width = 64
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
    
    def extrac_images(self):
        images_filename =  self.data_dir + 'images_normalized_' + str(self.size) + '_' + str(self.step) + '_' + str(self.n_samples_per_video) + '.npy'
        image_labels_filename = self.data_dir + 'images_labels_normalized_' + str(self.size) + '_' + str(self.step) + '_' + str(self.n_samples_per_video) + '.npy'

        if(not os.path.isfile(images_filename)):
            images = None
            labels = []
            folds = [x for x in os.listdir('data/') if x.startswith('Fold') and not '.zip' in x]

            for fold in folds:
                people = [x for x in os.listdir('data/'+fold) if x.isnumeric()]
                for person in people:
                    images_p_filename = self.data_dir + fold + '/' + person + '/images_normalized_' + str(self.height) + '_' + str(self.width) + '_' + str(self.n_samples_per_video) + '.npy'
                    image_labels_p_filename = self.data_dir + fold + '/' + person + '/images_labels_normalized_' + str(self.height) + '_' + str(self.width) + '_' + str(self.n_samples_per_video) + '.npy'
                    images_p = []
                    labels_p = []
                    if(not os.path.isfile(images_p_filename)):
                        bad = False
                        for i in [0,1]:
                            cap = cv2.VideoCapture(self.data_dir + fold + '/' + person + '/' + str(i) + '.mp4')
                            sec = 0
                            step = 1.0/self.frame_rate
                            success, image = get_frame(sec, cap)
                            count = 0
                            count_bads = 0
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            rotation = self.calculate_rotation(gray)
                            while success and count < self.n_samples_per_video:
                                if(rotation is not None):
                                    gray = cv2.rotate(gray, rotation)
                                
                                rects = self.detector(gray,0)
                                if(len(rects) >= 1):
                                    if(len(rects)==1):
                                        face = rects[0]
                                    else:
                                        face = extract_closest_face(rects)
                                    gray = gray[max(face.top(),0):max(face.bottom(),0), max(face.left(),0):max(face.right(),0)]
                                    gray = cv2.resize(gray, (self.height, self.width))
                                    gray = (gray-128)/128
                                    images_p.append(gray)
                                    count += 1
                                    labels_p.append([float(i)])
                                else:
                                    count_bads += 1
                                
                                if(count_bads > 100):
                                    bad = True
                                    break
                                sec = sec + step
                                sec = round(sec, 2)
                                success, image = get_frame(sec, cap)
                                if(success):
                                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            if(bad):
                                break
                        np.save(images_p_filename, np.array(images_p))
                        np.save(image_labels_p_filename, np.array(labels_p))
                    else:
                        images_p = np.load(images_p_filename, allow_pickle=True)
                        labels_p = np.load(image_labels_p_filename, allow_pickle=True)
                        count = len(images_p)
                    if(count == self.n_samples_per_video*2):
                        wall = int(len(images_p)/2)
                        images_0 = images_p[:wall]
                        images_1 = images_p[wall:]
                        labels_0 = labels_p[:wall]
                        labels_1 = labels_p[wall:]

                        vents_0, labels_0 = self.ventanizar(images_0, labels_0)
                        vents_1, labels_1 = self.ventanizar(images_1, labels_1)

                        images_p = np.concatenate((vents_0, vents_1), axis=0)
                        labels_p = np.concatenate((labels_0, labels_1), axis=0)


                        if(images is None):
                            images = images_p.copy()
                        else:
                            images = np.concatenate((images, images_p), axis=0)
                        labels.extend(labels_p)
                    else:
                        print("bad person:", fold, person)
            labels = np.array(labels)

            np.save(images_filename, images)
            np.save(image_labels_filename, labels)
        else:
            images = np.load(images_filename, allow_pickle=True)
            labels = np.load(image_labels_filename, allow_pickle=True)
        return images, labels

    def ventanizar(self, X, y):
        ventanas = []
        labels = []
        
        for i in range(0, len(X)-self.size, self.step):
            ventana = np.expand_dims(X[i:i+self.size], axis=-1)
            ventanas.append(ventana)
            labels_i = y[0]
            labels.append(labels_i)
        ventanas = np.array(ventanas)
        labels = np.array(labels)
        return ventanas, labels



