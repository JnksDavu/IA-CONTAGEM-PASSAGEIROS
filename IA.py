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

        # Flags para rastreamento das zonas
        self.passed_empty = False
        self.passed_entry = False
        self.passed_exit = False
        self.passed_right_empty = False  # Nova zona vazia à direita

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
        left_empty_zone_x = int(width * 0.15)   # Zona vazia (esquerda)
        entry_zone_x = int(width * 0.30)       # Zona de entrada (centro)
        exit_zone_x = int(width * 0.60)        # Zona de saída (direita)
        right_empty_zone_x = int(width * 0.85) # Nova zona vazia (à direita)

        # Desenhar as zonas no frame
        cv2.line(frame, (left_empty_zone_x, 0), (left_empty_zone_x, height), (0, 255, 255), 2)  # Amarelo (esquerda)
        cv2.line(frame, (entry_zone_x, 0), (entry_zone_x, height), (0, 255, 0), 2)              # Verde (entrada)
        cv2.line(frame, (exit_zone_x, 0), (exit_zone_x, height), (0, 0, 255), 2)               # Vermelho (saída)
        cv2.line(frame, (right_empty_zone_x, 0), (right_empty_zone_x, height), (0, 255, 255), 2) # Amarelo (direita)

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
                if center_x < left_empty_zone_x and not self.passed_empty:
                    print("[INFO] Objeto entrou na zona vazia (esquerda).")
                    self.passed_empty = True

                elif center_x > exit_zone_x and center_x < right_empty_zone_x and not self.passed_right_empty:
                    print("[INFO] Objeto entrou na zona vazia (direita).")
                    self.passed_right_empty = True

                elif center_x >= left_empty_zone_x and center_x < entry_zone_x and not self.passed_entry:
                    print("[INFO] Objeto entrou na zona de entrada.")
                    self.passed_entry = True

                elif center_x >= entry_zone_x and center_x < exit_zone_x and not self.passed_exit:
                    print("[INFO] Objeto entrou na zona de saída.")
                    self.passed_exit = True

                # Verifica a sequência e registra a entrada ou saída
                if self.passed_empty and self.passed_entry and self.passed_exit:
                    self.entry_count += 1
                    print(f"[EVENTO] Entrada detectada! Total: {self.entry_count}")
                    self._reset_flags()

                elif self.passed_right_empty and self.passed_exit and self.passed_entry:
                    self.exit_count += 1
                    print(f"[EVENTO] Saída detectada! Total: {self.exit_count}")
                    self._reset_flags()
            else:
                print("[WARNING] Rastreador perdeu o objeto. Reiniciando rastreador.")
                self.tracker = None
                self._reset_flags()

        # Exibir contagens na tela
        cv2.putText(frame, f"Entradas: {self.entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Saídas: {self.exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def _reset_flags(self):
        self.passed_empty = False
        self.passed_entry = False
        self.passed_exit = False
        self.passed_right_empty = False
