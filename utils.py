import numpy as np
import cv2
import math

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

### code adapted from the one created by ApurvaRaj in https://www.geeksforgeeks.org/find-angles-given-triangle/
# returns square of distance b/w two points  
def lengthSquare(X, Y):  
    xDiff = X[0] - Y[0]  
    yDiff = X[1] - Y[1]  
    return xDiff * xDiff + yDiff * yDiff 
      
def get_cosines(A, B, C):
    if(A[0] == B[0] and A[1] == B[1]):
        if(B[0]+1 == C[0] and B[1]==C[1]):
            B[0] = B[0] + 2
        else:
            B[0] = B[0] + 2
    if(C[0] == B[0] and C[1] == B[1]):
        if(B[0]+1==A[0] and B[1]==A[1]):
            B[0] = B[0] + 2
        else:
            B[0] = B[0] + 1
    if(A[0] == C[0] and A[1] == C[1]):
        if(C[0]+1 == B[0] and C[1]==B[1]):
            C[0] = C[0] + 2
        else:
            C[0] = C[0] + 1
      
    # Square of lengths be a2, b2, c2
    a2 = lengthSquare(B, C)  
    b2 = lengthSquare(A, C)  
    c2 = lengthSquare(A, B)  
  
    # length of sides be a, b, c
    a = math.sqrt(a2);  
    b = math.sqrt(b2);  
    c = math.sqrt(c2);  
  
    # From Cosine law
    alpha = math.acos(min(max((b2 + c2 - a2) / (2 * b * c), -1),1))
    betta = math.acos(min(max((a2 + c2 - b2) / (2 * a * c), -1),1))
    gamma = math.acos(min(max((a2 + b2 - c2) / (2 * a * b), -1),1))
  
    # Converting to degree  
    alpha = alpha * 180 / math.pi
    betta = betta * 180 / math.pi
    gamma = gamma * 180 / math.pi

    # # applying cosine transformation
    # alpha = np.cos(np.deg2rad(alpha))
    # betta = np.cos(np.deg2rad(betta))
    # gamma = np.cos(np.deg2rad(gamma))

    return alpha, betta, gamma
### end code adapted from ApurvaRaj

def get_useful_triangles(ls):
    triangles = []
    # triangle coords manually extracted from paper: A fuzzy logic approach to reliable real-time recognition of facial emotion.
    triangles_coords = [[17,19,36], [17,21,37], [19,21,39], [19,36,39], [36,37,40], [38,39,41],
                        [22,24,42], [22,26,44], [24,26,45], [24,42,45], [42,43,46], [43,45,47],
                        [21,22,27], [36,48,60], [39,42,51], [45,54,64], [48,51,54], [48,54,57]]
    for a,b,c in triangles_coords:
        triangles.append([ls[a], ls[b], ls[c]])

    return triangles

def extract_closest_face(faces):
    areas = []
    for f in faces:
        areas.append(f.height() * f.width())
    
    idx = np.argmax(areas)
    return faces[idx]