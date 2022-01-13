import cv2
'''
def facial_recognition(face_image):
    # face recognition
    
    #Computes the PCA of the face
    faceTestPCA = pca.transform(face.reshape(1, -1))
    pred = clf.predict(faceTestPCA)

    print(names[pred[0]])

    cv2.putText(img, str(names[pred[0]]), (x+5,y-5), font, 1, (255,255,255) , 2)
'''
def face_detect(img,face_cascade):
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Documentation: https://realpython.com/face-recognition-with-python/
    # https://www.bogotobogo.com/python/ONo pet or pet is dead. Use /start <name> to create a new pet!
    # OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
    # Pre-trained weights: https://github.com/opencv/opencv/tree/master/data/haarcascades

    # Detect faces # detection algorithm uses a moving window to detect objects
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, # Since some faces may be closer to the camera, they would appear bigger than the faces in the back. The scale factor compensates for this.
                                    )
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        #draw rectangles where face is detected
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        #Extracts the face from grayimg, resizes and flattens
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200,200))
        #face = face.ravel()

    return img, faces