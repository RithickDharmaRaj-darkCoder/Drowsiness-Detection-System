import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer

mixer.init()
mixer.music.load("music.wav")

# Eye Aspect Ratio
def e_a_r(eye):
	A = distance.euclidean(eye[1], eye[5])             # Verticle 1st joint
	B = distance.euclidean(eye[2], eye[4])             # Verticle 2st joint
	C = distance.euclidean(eye[0], eye[3])             # Horizontal joint
	ear = (A + B) / (2.0 * C)
	return ear

cap=cv2.VideoCapture(0)
count = 0
frame_check = 20        # Minimum time to initiate Alarm
thresh = 0.25           # Try (0.3 to 0.45) if 0.25 not works properly [Eyebrows gap]
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")     # Imported Face shape in dotted patterns (68 dots).

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray_frame, 0)
 
	for subject in subjects:
		shape = predict(gray_frame, subject)
		shape = face_utils.shape_to_np(shape)
  
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
  
		leftEAR = e_a_r(leftEye)
		rightEAR = e_a_r(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
  
        # To draw border abound eye.
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
  
		if ear < thresh:
			count += 1
			print (count)
   
			if count >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				mixer.music.play()
		else:
			count = 0
   
	cv2.imshow("WebCam Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break

cv2.destroyAllWindows()
cap.release()
