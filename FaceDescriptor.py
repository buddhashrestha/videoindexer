import dlib
import cv2
import numpy
import os

p = "./dlib-models/shape_predictor_68_face_landmarks.dat"
landmarks = "./dlib-models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


facerec = dlib.face_recognition_model_v1(
    "./dlib-models/dlib_face_recognition_resnet_model_v1.dat")


class FaceDescriptor(object):

    def __init__(self, image_path):
        self.image_path = image_path

    def getDescriptor(self):
        if os.path.isfile(self.image_path):
            #File is present
            print("processing: ", self.image_path)
            image = cv2.imread(self.image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for (i, rect) in enumerate(rects):
                landmark = dlib.shape_predictor(landmarks)(image, rect)
                face_descriptor = facerec.compute_face_descriptor(
                    image, landmark)
            print("Done")
            # this doesn't work for no face in the photo
            return numpy.array(face_descriptor)

        # Return this string if file is not present.
        else:
            print("File not found", self.image_path)
            return "File not found."


print(FaceDescriptor('/home/buddha/Desktop/marisa.jpg').getDescriptor())
