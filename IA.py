import cv2
import numpy as np
import time

class PersonAndFaceDetector:
    def __init__(self):
        self.face_net = cv2.dnn.readNetFromCaffe(
            'models/deploy.prototxt',
            'models/res10_300x300_ssd_iter_140000.caffemodel'
        )

        self.tracker = None
        self.last_event_time = 0
        self.event_cooldown = 3  # Intervalo mínimo entre eventos
        self.crossed_empty = False
        self.crossed_entry = False
        self.crossed_exit = False
        self.passed_empty = False  # Marca se o objeto passou pela zona vazia
        self.passed_entry = False  # Marca se o objeto passou pela zona de entrada
        self.passed_exit = False   # Marca se o objeto passou pela zona de saída

        # Variáveis para contagem
        self.entry_count = 0
        self.exit_count = 0

    def detect_faces_dnn(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Confiança mínima
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                return (startX, startY, endX - startX, endY - startY)
        return None

    def classify_scene(self, frame):
        height, width = frame.shape[:2]
        
        # Definir as zonas
        empty_zone_x = int(width * 0.15)  # Zona vazia à esquerda
        entry_zone_x = int(width * 0.50)  # Zona de entrada no centro
        exit_zone_x = int(width * 0.85)   # Zona de saída à direita

        # Desenhar as zonas na tela
        cv2.line(frame, (empty_zone_x, 0), (empty_zone_x, height), (0, 255, 255), 2)  # Amarelo (Zona Vazia)
        cv2.line(frame, (entry_zone_x, 0), (entry_zone_x, height), (0, 255, 0), 2)    # Verde (Entrada)
        cv2.line(frame, (exit_zone_x, 0), (exit_zone_x, height), (0, 0, 255), 2)     # Vermelho (Saída)

        if self.tracker is None:
            face_box = self.detect_faces_dnn(frame)
            if face_box:
                print("[INFO] Objeto detectado. Inicializando o rastreador.")
                self.tracker = cv2.TrackerCSRT_create()
                self.tracker.init(frame, tuple(face_box))
            else:
                print("[WARNING] Nenhum objeto detectado no frame.")
        else:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                center_x = x + w // 2

                # Desenha a caixa ao redor da pessoa detectada
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Atualizar estados com base na posição
                if center_x < empty_zone_x and not self.passed_empty:
                    print("[INFO] Objeto entrou na zona vazia.")
                    self.passed_empty = True  # Marca que a pessoa passou pela zona vazia

                elif center_x >= empty_zone_x and center_x < entry_zone_x and not self.passed_entry:
                    print("[INFO] Objeto entrou na zona de entrada.")
                    self.passed_entry = True  # Marca que a pessoa passou pela zona de entrada

                elif center_x >= entry_zone_x and center_x < exit_zone_x and not self.passed_exit:
                    print("[INFO] Objeto entrou na zona de saída.")
                    self.passed_exit = True  # Marca que a pessoa passou pela zona de saída

                # Verifica a sequência e registra a entrada ou saída
                if self.passed_empty and self.passed_entry and self.passed_exit:
                    # Se o objeto passou pela zona vazia, depois pela zona de entrada e por fim pela zona de saída
                    self.entry_count += 1
                    print(f"[EVENTO] Entrada detectada! Total: {self.entry_count}")
                    # Reseta os estados após registrar a entrada
                    self.passed_empty = False
                    self.passed_entry = False
                    self.passed_exit = False
                    self.last_event_time = time.time()

                elif self.passed_empty and self.passed_exit and self.passed_entry:
                    # Se o objeto passou pela zona vazia, depois pela zona de saída e por fim pela zona de entrada
                    self.exit_count += 1
                    print(f"[EVENTO] Saída detectada! Total: {self.exit_count}")
                    # Reseta os estados após registrar a saída
                    self.passed_empty = False
                    self.passed_entry = False
                    self.passed_exit = False
                    self.last_event_time = time.time()

            else:
                print("[WARNING] Rastreador perdeu o objeto. Reiniciando rastreador.")
                self.tracker = None
                self.passed_empty = False
                self.passed_entry = False
                self.passed_exit = False

        # Exibir contagens na tela
        cv2.putText(frame, f"Entradas: {self.entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Saídas: {self.exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame
