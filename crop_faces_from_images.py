import cv2
import numpy as np
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open a sample video available in sample-videos
video = cv2.VideoCapture(0)
width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)

id = "zhili"
img_directory = f"collected-images/{id}"
crop_directory = "face-database"

img_num = 1

if not os.path.exists(f"{crop_directory}/{id}"):
    os.makedirs(f"{crop_directory}/{id}")

dirs = os.listdir(img_directory)
for i,img_path in enumerate(dirs):
    print(img_path)
    # Capture frame-by-frame

    frame = cv2.imread(img_directory+'/'+img_path)

    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    #cv2.imshow('frame',frame)

    # Detect faces # detection algorithm uses a moving window to detect objects
    faces = face_cascade.detectMultiScale(gray, 
                                    scaleFactor=1.1, # Since some faces may be closer to the camera, they would appear bigger than the faces in the back. The scale factor compensates for this.
                                    minNeighbors=5 
                                    )

    count = 0
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        
        #Extracts the face from grayimg, resizes and flattens
        face_gray = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_gray, (200,200))
        cv2.imwrite(f"{crop_directory}/{id}/{i}_{count}.jpg",face_resized)
        #face = face.ravel() # returns flattened array
        count+= 1
    
    #if cv2.waitKey(10) & 0xFF == ord('q'):
            #break

#cv2.destroyAllWindows()


 
    