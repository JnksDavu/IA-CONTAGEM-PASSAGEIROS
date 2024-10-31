import cv2
import face_recognition
import numpy as np

# Inicialize a lista para armazenar as características faciais únicas
known_face_encodings = []
unique_faces_count = 0

# Captura de vídeo (0 para webcam ou o caminho do arquivo de vídeo)
video_capture = cv2.VideoCapture("video.mp4")  # ou 0 para webcam

while video_capture.isOpened():
    # Ler um quadro do vídeo
    ret, frame = video_capture.read()
    if not ret:
        break

    # Redimensionar o quadro para processamento mais rápido
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Localizar e codificar rostos no quadro atual
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_encoding in face_encodings:
        # Verifique se o rosto atual é semelhante a um rosto conhecido
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        if not any(matches):
            # Novo rosto encontrado; adicione às codificações conhecidas
            known_face_encodings.append(face_encoding)
            unique_faces_count += 1

    # Exibir o número de pessoas únicas identificadas no vídeo
    cv2.putText(frame, f"Pessoas únicas: {unique_faces_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Exibir o quadro de vídeo
    cv2.imshow('Video', frame)

    # Parar com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpar
video_capture.release()
cv2.destroyAllWindows()
