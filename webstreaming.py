# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import json


from src.faceDetection import fdetect
import src.predict as predict
import src.constants as const

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
outputFrame2 = None
outputAttributes= None
presence=False
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_face(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, outputFrame2, outputAttributes, lock, presence

	# loop over frames from the video stream
	while True:
		
		# read the next frame from the video stream and resize it
		frame = vs.read()

		center_w, center_h = (const.FRAME_WIDTH//2, const.FRAME_HEIGHT//2)
		frame = imutils.resize(frame, width=const.FRAME_WIDTH,height=const.FRAME_HEIGHT)
		

		frame_cut=frame.copy()[center_h-109:center_h+109,center_w-89:center_w+89]

		# grab the current timestamp and draw it on the frame
		# timestamp = datetime.datetime.now()
		# cv2.putText(frame, timestamp.strftime(
		# 	"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		faces= fdetect(frame_cut)

	#Draw a ellipse to center de face
		axesLength = (80, 100) 
		angle = 0
		startAngle = 0
		endAngle = 360
		#if there is a face the elipse is green, else red
		if faces[0]:
			color = (0, 255,0)
		else:
			color = (0, 0, 255)
		# Line thickness of 5 px 
		thickness = 5
 
		cv2.ellipse(frame, (center_w, center_h), axesLength, 
		 		angle, startAngle, endAngle, color, thickness) 
		frame=cv2.flip(frame,+1)

		attributes=predict.predict(faces[0], frame_cut)
 
		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy() 
			outputFrame2 = frame_cut.copy()
			presence=faces[0]
			outputAttributes = attributes.copy()


def ImgGenerate(mode='function'):
	# grab global references to the output frame and lock variables
	global outputFrame,outputFrame2, lock

	# loop over frames from the output stream
	while True:
		
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			if mode=='debug':
				(flag, encodedImage) = cv2.imencode(".jpg", outputFrame2)
			else:
				(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

def attGenerate():
	global outputAttributes, lock

	while True:
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputAttributes is None:
				continue

		# yield the output
		#return json.dumps(outputAttributes)
		#print(outputAttributes)
		return json.dumps(outputAttributes) 

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(ImgGenerate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed2")
def video_feed2():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(ImgGenerate('debug'),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/attributes")
def attributes():
	# return the response generated along with the specific media
	# type (mime type)

	return Response(attGenerate(), mimetype='application/json')



# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection


	#get_default_graph()
	t1 = threading.Thread(target=detect_face, args=(
		args["frame_count"],))
	t1.daemon = True
	t1.start()


	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()