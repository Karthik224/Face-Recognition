from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import datetime
import cv2
import dlib
from PIL import Image
import os
import sys
import time
import xlsxwriter
import csv
os.system("python3 att_excel.py")
count ={}
countUnknown =0
def detect_faces(image):
    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()
    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]
    return face_frames
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
args = vars(ap.parse_args())
# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
# start the FPS counter
fps = FPS().start()
# loop over frames from the video file stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"
		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		# update the list of names
		names.append(name)
	# loop over the recognized faces
	for((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		if name =='Unknown':
                     rgbImage = frame
                     faces_detected = detect_faces(rgbImage)
                     for n,face_rects in enumerate(faces_detected):
                        face = Image.fromarray(rgbImage).crop(face_rects)
                        b,g,r = face.split()
                        face = Image.merge('RGB',(r,g,b))
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
                        now = datetime.datetime.now()
                        image_name = output_dir + '/'+ person+'{}'.format(str(count[person])+now.strftime("%H:%M:%S"))+'.jpg'
                        face.save(image_name)
                        os.system('pkill display')
                        countUnknown +=1
                        if countUnknown == 10:   
                            os.system('python3 encode_faces.py --encodings encodings.pickle --dataset dataset --detection-method hog')
                        #os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)
                            sys.exit(0)
		#cv2.putText(frame,str(counts[name]+1),(right,y+1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
		cv2.putText(frame,name,(right,y),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,0,0),2)
	# display the image to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()

