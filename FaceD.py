import cv2
import os
print("OpenCV version:", cv2.__version__)

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('Enter face id: ')
print("\n [INFO] Initializing...")
count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.5, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite("Dataset/User." + str(face_id) + '.' + str(count) + '.jpg', gray[y:y + h, x:x + w])
    img = cv2.flip(img, 1)
    cv2.imshow('Camera Feed', img)

    k = cv2.waitKey(100)
    if k == 27:
        break
    elif count >= 500:
        break

print("\n [INFO] Exiting...")
cam.release()
cv2.destroyAllWindows()
