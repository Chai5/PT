import cv2

Algoritmo_dr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

Img_t = cv2.imread('El jony.png', 0)

# image = cv2.imread('El jony.png')

# Imagen a escala de grises
Img_g = cv2.cvtColor(Img_t, cv2.COLOR_BGR2RGB)

# Deteccion del rostro de la imagen
Deteccion_r = Algoritmo_dr.detectMultiScale(Img_g, 1.32, 5)


for (x,y,w,h) in Deteccion_r:
  cv2.rectangle(Img_t,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow('image',Img_t)
cv2.waitKey(0)
cv2.destroyAllWindows()

