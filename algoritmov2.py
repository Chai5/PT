import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model


# Asigancion del modelo entrenado
Modelo = load_model("modelo2.h5")

# Asigasion de algoritmo para deteccion de rostros
Algoritmo_dr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Habilita la camara de la computadora
cap = cv2.VideoCapture(0)

while True:
    # Captura el cuadro y devuelve el valor booleano y la imagen capturada
    ret, Img_t = cap.read()
    if not ret:
        continue
    # Imagen a escala de grises
    Img_g = cv2.cvtColor(Img_t, cv2.COLOR_BGR2RGB)

    # Deteccion del rostro de frente a la camara
    Deteccion_r = Algoritmo_dr.detectMultiScale(Img_g, 1.32, 5)

    for (x, y, w, h) in Deteccion_r:
        cv2.rectangle(Img_t, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = Img_g[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        Prediccion = Modelo.predict(img_pixels)

        Max_index = np.argmax(Prediccion[0])

        Emociones = ('Enojo', 'Felicidad', 'Tristeza', 'Neutral')

        if Max_index == 1:
            Max_index = 0
        if Max_index == 3 or Max_index == 5:
            Max_index = 1
        if Max_index == 4:
            Max_index = 2
        if Max_index == 6:
            Max_index = 4

        # Prediccion de la emocion
        Prediccion_e = Emociones[Max_index]
        print(Emociones[Max_index])

        cv2.putText(Img_t, Prediccion_e, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(Img_t, (1000, 700))
    cv2.imshow('Analisis de emocion facial', resized_img)

    if cv2.waitKey(10) == ord('x'):
        break

cap.release()
cv2.destroyAllWindows