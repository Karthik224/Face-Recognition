import cv2
import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import os
import time
import datetime
now=datetime.datetime.now()
def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames
def CaptureImage():
    imageName = ''
    cap = cv2.VideoCapture(0)
    count = {}
    count1 =0
    required = int(input("Input the number of images you want to save:"))                   
    output_dir = input('Enter the directory where you want to save the images:')
    a =[]
    if os.path.exists(output_dir):
        pass
    else:
        list1 = output_dir.split('/')
        name = ''
        for i in list1:
            name += i
            name += '/'
            if os.path.exists(name):
                pass
            else:
                os.mkdir(name)
    while True:
        ret,frame = cap.read()
        rgbImage = frame
        cv2.imshow('WebCam',rgbImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detected_faces=detect_faces(rgbImage)
            for n,face_rect in enumerate(detected_faces):
                face = Image.fromarray(rgbImage).crop(face_rect)
                b,g,r = face.split()#splitting the face parts basing the colour order b,g,r
                face = Image.merge('RGB',(r,g,b))#merging the those face parts that divided in the order r,g,b
                face.show()
                person = input('Enter the person name:')
                output_dir = 'dataset/'+person
                if os.path.exists(output_dir):
                    pass
                else:
                    output_dir_list = output_dir.split('/')
                    name =''
                    for i in output_dir_list:
                        name += i+'/'
                        if os.path.exists(name):
                            pass
                        else:
                            os.mkdir(name)
                if person in count.keys():
                    count[person] +=1
                else:
                    count[person] = 0
                image_name = output_dir + '/'+ person+'{}'.format(str(count[person])+now.strftime("%H:%M:%S"))+'.jpg'
                face.save(image_name)
                os.system('pkill display')
            if count1 == required - 1:
                break
            else:
                count1 +=1
    cap.release()
    cv2.destroyAllWindows()
    return a
CaptureImage()

