import cv2
import numpy as np
from keras.models import model_from_json

expression_dict = {0: "MARAH", 1: "JIJIK", 2: "TAKUT", 3: "SENANG", 4: "NETRAL", 5: "SEDIH", 6: "TERKEJUT"}

json_file = open('expression_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
expression_model = model_from_json(loaded_model_json)

expression_model.load_weights("expression_model.h5")
print("Model berhasil dimuat")

#menggunakan webcam
cap = cv2.VideoCapture(0)

#menggunakan file video
#cap = cv2.VideoCapture("D:\\Kuliah_Teknik_Informatika\\SEMESTER_4\\Kecerdasan_Buatan\\PROJECT\\expression_CNN\\sample.mp4")


while True:
    # menggunakan haarcascade untuk menggambar kotak di sekitar wajah
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # deteksi wajah pada kamera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # preprocessing wajah yang terdapat pada kamera
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # prediksi ekspresi
        expression_prediction = expression_model.predict(cropped_img)
        maxindex = int(np.argmax(expression_prediction))
        cv2.putText(frame, expression_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Deteksi Ekspresi', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
