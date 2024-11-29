import cv2
from Camera import Camera
from IA import PersonAndFaceDetector

def main():
    camera = Camera()
    detector = PersonAndFaceDetector() 

    print("Iniciando a detecção de entradas e saídas. Pressione 'q' para encerrar.")

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Erro ao capturar frame. Encerrando...")
            break

        classification = detector.classify_scene(frame)
        cv2.imshow('Contagem de Passageiros', classification)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Encerrando o programa...")
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
