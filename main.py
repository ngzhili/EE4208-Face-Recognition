
import cv2
import numpy as np
#from PIL import Image
import os

from face_utils import face_detect

''' ========== Facial Recognition ========== '''
# create database
face_vector = []
face_path = []

img_directory = f"face-database/data"
dirs = os.listdir(img_directory)

for i,img_path in enumerate(dirs):
    #print(img_path)
    # Capture frame-by-frame
    frame = cv2.imread(img_directory+'/'+img_path)
    frame = cv2.resize(frame,(100,100))
    #print(frame.shape)
    image_width,image_length,_ = frame.shape
    total_pixels = image_width*image_length

    face_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face_image = face_image.reshape(total_pixels,)
    face_vector.append(face_image)
    face_path.append(img_path)
#print(face_path)

face_vector = np.asarray(face_vector)
face_vector = face_vector.transpose()

# noramlizing face vectors
avg_face_vector = face_vector.mean(axis=1)
avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)
normalized_face_vector = face_vector - avg_face_vector
#print(normalized_face_vector)

# calculate co-variance matrix
covariance_matrix = np.cov(np.transpose(normalized_face_vector))
#print(covariance_matrix)

# calculate eigen values and eigen vectors
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
#print(eigen_vectors.shape)

#Select the K best Eigen Faces, K < M
k = 20
k_eigen_vectors = eigen_vectors[0:k, :]
#print(k_eigen_vectors.shape)

# Convert lower dimensional K Eigen Vectors to Original Dimension
eigen_faces = k_eigen_vectors.dot(normalized_face_vector.T)
#print(eigen_faces.shape)

# STEP7: Represent Each eigen face as combination of the K Eigen Vectors
# weights = eigen_faces.dot(normalized_face_vector)
weights = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))


''' ========== Face Detection ========== '''
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open a sample video available in sample-videos
video = cv2.VideoCapture(0)
width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)

font = cv2.FONT_HERSHEY_SIMPLEX
#fourcc = 0x31637661
#videoWriter = cv2.VideoWriter(f'computer_vision/cv-images/{group_id}_video_temp.mp4', fourcc, fps, (int(width), int(height)))

''' ========== Run Live WebCam Inference ========== '''
while(True):
    # Capture frame-by-frame

    ret, frame = video.read()
    if not ret:
        print("failed to grab frame")
        break

    # run face detection
    gray_image, faces = face_detect(frame,face_cascade)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        #draw rectangles where face is detected
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        #Extracts the face from grayimg, resizes and flattens
        face = gray_image[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (100,100))
        
        #face = face.ravel() # returns flattened array
        #test_img = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)

        image_width,image_length = face_resized.shape
        total_pixels = image_width*image_length

        face_resized = face_resized.reshape(total_pixels, 1)
        test_normalized_face_vector = face_resized - avg_face_vector
        test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))

        index =  np.argmin(np.linalg.norm(test_weight - weights, axis=1))    
        #print('Predicted index:',index)
        #print('Test Face:',test_add.split('/')[-1].split('_')[0])
        print('Predicted Face:',face_path[index].split('_')[0])
        cv2.putText(frame, face_path[index].split('_')[0], (x+5,y-5), font, 1, (255,255,255) , 2)

    cv2.putText(frame, f'Number of Faces Detected: {len(faces)}', (10,20), font, 0.7, (255,255,255) , 2)
            
    # Display the output in window
    cv2.imshow('face detection', frame)
    #videoWriter.write(processed_img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
#videoWriter.release()
