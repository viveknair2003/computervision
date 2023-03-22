import cv2

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#the parameter is 0 because we are going to use only one internal camera here to capture
camera = cv2.VideoCapture(0)


while True:
    success, img = camera.read()
    output = face_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10)
    for x, y, w, h in output :
        cv2.rectangle(img, (x, y), (x + w, y + h), (333, 0, 111), 3)

    cv2.imshow("camera", img)
    key=cv2.waitKey(10)
    if key==32:      #ascii value of spacbar is 32 .
        break;

#To release your resources. It is a good practice to add these!
camera.release()
cv2.destroyAllWindows()