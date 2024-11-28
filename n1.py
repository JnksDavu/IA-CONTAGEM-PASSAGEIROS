import cv2

# Carregar o classificador de rosto pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Abrir o vídeo
video_path = 'Video.mp4'  # Substitua pelo caminho do seu vídeo
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Sai do loop se não houver mais quadros

    # Converter o quadro para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10)

    # Desenhar retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar o quadro processado
    cv2.imshow('Face Detection', frame)

    # Sair do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar o objeto de captura e fechar as janelas
cap.release()
cv2.destroyAllWindows()
