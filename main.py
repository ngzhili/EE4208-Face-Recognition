
import cv2
import numpy as np
#from PIL import Image
import os

from face_utils import face_detect

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open a sample video available in sample-videos
video = cv2.VideoCapture(0)
width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)

#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#fourcc = cv2.VideoWriter_fourcc(*'H264')
#fourcc = 0x31637661
#fourcc = cv2.VideoWriter_fourcc(*'X264')
#fourcc = cv2.VideoWriter_fourcc(*'avc1')
#fourcc = 0x31637661
#videoWriter = cv2.VideoWriter(f'computer_vision/cv-images/{group_id}_video_temp.mp4', fourcc, fps, (int(width), int(height)))
#videoWriter = cv2.VideoWriter(f'computer_vision/cv-images/{group_id}_video_temp.mp4', fourcc, fps, (int(width), int(height)))

#prediction_count = 0

while(True):
    # Capture frame-by-frame

    ret, frame = video.read()
    if not ret:
        print("failed to grab frame")
        break
    
    # Display the resulting frame
    #cv2.imshow('frame',frame)

    # run face detection
    processed_img, predictions = face_detect(frame,face_cascade)

    #videoWriter.write(processed_img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(processed_img, f'Number of Faces Detected: {len(predictions)}', (100,100), font, 1, (255,255,255) , 2)
            
    #cv2.putText(processed_img, f'Number of Faces Deteced: {len(predictions)}', (x+5,y-5), font, 1, (255,255,255) , 2)
    
    # Display the output in window
    cv2.imshow('face detection', processed_img)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
#videoWriter.release()
