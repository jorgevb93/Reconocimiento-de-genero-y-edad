# Importamos las librerias
from deepface import DeepFace
import cv2
import mediapipe as mp

# Declaramos la deteccion de rostros
detros = mp.solutions.face_detection
rostros = detros.FaceDetection(min_detection_confidence= 0.8, model_selection=0)
mp_drawing = mp.solutions.drawing_utils

# Realizamos VideoCaptura
cap = cv2.VideoCapture(0)

# Empezamos
while True:
    # Leemos los fotogramas
    ret, frame = cap.read()

    # Correccion de color
    frame=cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesamos
    resrostros = rostros.process(rgb)

    # Deteccion
    if resrostros.detections is not None:
        # Registramos
        for rostro in resrostros.detections:

            mp_drawing.draw_detection(frame, rostro)
        
            # Informacion
            info = DeepFace.analyze(rgb, actions=['age', 'gender'], enforce_detection= False)

            # Edad
            edad = info['age']

            # Genero
            gen = info['gender']

            # Traducimos
            if gen == 'Man':
                gen = 'Hombre'

            elif gen == 'Woman':
                gen = 'Mujer'

            # Mostramos info
            cv2.putText(frame, str(gen), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(edad), (500, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
# Mostramos los fotogramas
    cv2.imshow(" Deteccion", frame)
    
    # Leemos el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()
