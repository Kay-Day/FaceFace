from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['Luong Thuy Vy', 'Le Dung', 'Nguyen Trung']

person_info_dict = {
    'Luong Thuy Vy': {
        'name': 'Luong Thuy Vy',
        'age': 25,
        'address': '123 Street, City',
        'occupation': 'Software Engineer'
    },
    'Le Dung': {
        'name': 'Le Dung',
        'age': 30,
        'address': '456 Street, Town',
        'occupation': 'Data Scientist'
    },
    'Nguyen Trung': {
        'name': 'Nguyen Trung',
        'age': 61,
        'address': '456 Street, Town',
        'occupation': 'Data Scientist'
    },
}

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

def generate_frames():
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 100:
                person_name = names[id]
                person_info = person_info_dict.get(person_name, {})
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                person_name = "Unknown"
                person_info = {}
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, f"Person: {person_name}", (x + 5, y - 25), font, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Age: {person_info.get('age', 'Unknown')}", (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Address: {person_info.get('address', 'Unknown')}", (x + 5, y + 15), font, 1,
                        (255, 255, 255), 2)
            cv2.putText(img, f"Occupation: {person_info.get('occupation', 'Unknown')}", (x + 5, y + 35), font, 1,
                        (255, 255, 255), 2)
            cv2.putText(img, f"Confidence: {confidence}", (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        ret, jpeg = cv2.imencode('.jpg', img)
        data = []
        if ret:
            data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
