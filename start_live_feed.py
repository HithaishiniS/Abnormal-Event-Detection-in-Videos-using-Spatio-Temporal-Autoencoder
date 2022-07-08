
'''
Running The Spatiotemporal autoencoder on live webcam field
run python3 start_live_feed.py 'path_to_model' to start the feed and processing

'''

import cv2
from model import load_model
import numpy as np 
from skimage.transform import resize 
from test import mean_squared_loss
from keras.models import load_model
import argparse
import os
import glob
import time
#from PIL import Image as im
import winsound

frame_no=0
def remove_old_images(path):
	filelist = glob.glob(os.path.join(path, "*.jpg"))
	for f in filelist:
		os.remove(f)

parser=argparse.ArgumentParser()

parser.add_argument('modelpath',type=str)

args=parser.parse_args()

modelpath=args.modelpath

#Remove old images
remove_old_images('./livefeed')


vc=cv2.VideoCapture(0)
currentframe=1
rval=True
print('Loading model')
model=load_model(modelpath)
print('Model loaded')

if not os.path.exists('livefeed'):
	os.makedirs('livefeed')


threshold=0.00072
while True:
	imagedump=[]
	
	for i in range(10):
		rval,frame=vc.read()
		cv2.imshow("Output",frame)	
		key = cv2.waitKey(1)
		cv2.imwrite('./livefeed/liveframe'+str(currentframe)+'.jpg',frame)


		frame=resize(frame,(227,227,3))

		#Convert the Image to Grayscale


		gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
		gray=(gray-gray.mean())/gray.std()
		gray=np.clip(gray,0,1)
		imagedump.append(gray)
		currentframe+=1
	
	
	

	imagedump=np.array(imagedump)

	imagedump.resize(227,227,10)
	imagedump=np.expand_dims(imagedump,axis=0)
	imagedump=np.expand_dims(imagedump,axis=4)


	print('Processing data')

	output=model.predict(imagedump)
	frame_no=frame_no+1



	loss=mean_squared_loss(imagedump,output)


	if loss>threshold:
		print('Anomalies Detected')
		frame_startno=((frame_no-1)*10)+1
		frame_stopno=frame_startno+9
		print("Anamolous frames detected from frame {} to frame {}".format(frame_startno,frame_stopno))
		winsound.Beep(500,1000)
	
		
	else:
		print("normal")


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
