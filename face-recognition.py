import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
# https://reubenrochesingh.medium.com/building-face-detector-using-principal-component-analysis-pca-from-scratch-in-python-1e57369b8fc5
# https://analyticsindiamag.com/10-face-datasets-to-start-facial-recognition-projects/

# create database
face_vector = []
face_path = []

img_directory = f"face-database/data"

dirs = os.listdir(img_directory)

#count = 0
for i,img_path in enumerate(dirs):
    #print(img_path)
    # Capture frame-by-frame
    #if count <= 20:
    frame = cv2.imread(img_directory+'/'+img_path)
    frame = cv2.resize(frame,(100,100))
    #print(frame.shape)
    image_width,image_length,_ = frame.shape
    total_pixels = image_width*image_length

    face_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face_image = face_image.reshape(total_pixels,)
    face_vector.append(face_image)
    face_path.append(img_path)
    #count+=1 
    #else:
        #break
#print(face_path)

face_vector = np.asarray(face_vector)
face_vector = face_vector.transpose()
#print(face_vector)
#cv2.imshow('image',face_vector[0])
#cv2.waitKey(1000) 
#print(face_vector.shape)
#print(face_vector)


# normalizing face vectors
avg_face_vector = face_vector.mean(axis=1)
avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)
normalized_face_vector = face_vector - avg_face_vector
#print(normalized_face_vector)


#'''
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

# Represent Each eigen face as combination of the K Eigen Vectors
# weights = eigen_faces.dot(normalized_face_vector)
weights = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))
#print(weights)

#'''
#STEP8: Testing Phase
test_add = "face-database/data/s1_5.jpg" # "testing/" + "8" + ".jpg"
test_add = "face-database/data/s12_6.jpg" 
test_add = "face-database/data/s23_9.jpg" 
test_add = "face-database/zhili_2.jpg"
test_add = "face-database/data/s3_6.jpg" 

test_img = cv2.imread(test_add)
test_img = cv2.resize(test_img,(100,100))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

image_width,image_length = test_img.shape
total_pixels = image_width*image_length
test_img = test_img.reshape(total_pixels, 1)
test_normalized_face_vector = test_img - avg_face_vector
test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))

index =  np.argmin(np.linalg.norm(test_weight - weights, axis=1))    
print('Predicted index:',index)
print('Test Face:',test_add.split('/')[-1].split('_')[0])
print('Predicted Face:',face_path[index].split('_')[0])
'''     
if(index>=0 and index <5):
    print("Nandan Raj")
if(index>=5 and index<10):
    print("Saurabh Mishra")
if(index>=10 and index<15):
    print("Gagan Ganpathy")
if(index>=15 and index<20):
    print("Badnena Upendra")
if(index>=20 and index<25):
    print("Sai Charan")
if(index>=25 and index<30):
    print("Luv NA")
if(index>=30 and index<35):
    print("Manavdeep Singh")
if(index>=35 and index<40):
    print("Anagh Rao")
#'''