import cv2

# Constantes de configuração
CAT_CLASS_ID = 8
CONFIDENCE_THRESHOLD = 0.3
BLOB_SCALE_FACTOR = 1 / 127.5
BLOB_SIZE = (300, 300)
BLOB_MEAN = (127.5, 127.5, 127.5)
FRAME_COUNT_THRESHOLD = 4
LASER_OFFSET = 50
LASER_MOVEMENT_MULTIPLIER = 2

# Modelos DNN
PROTO_TXT = "model/MobileNetSSD_deploy.prototxt"
CAFFE_MODEL = "model/MobileNetSSD_deploy.caffemodel"

DNN = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)

# Inicialização de captura de vídeo e rastreador
cap = cv2.VideoCapture("data/3.mp4")
tracker = cv2.TrackerKCF_create()
tracking = False

prev_centers = []

def draw_laser(frame, cat_edges, movement_vector, direction, offset):
    """
    Desenha o ponto do laser na tela baseado no movimento do gato.
    """
    height, width, _ = frame.shape
    cat_left_edge, cat_right_edge, cat_bottom_edge = cat_edges

    if direction == "right":
        laser_x = cat_right_edge + (offset * movement_vector[0]) + LASER_OFFSET
        laser_y = cat_bottom_edge + movement_vector[1]
        if laser_x > width:
            laser_x = cat_left_edge - (offset * movement_vector[0]) - LASER_OFFSET
    else:
        laser_x = cat_left_edge - (offset * movement_vector[0]) - LASER_OFFSET
        laser_y = cat_bottom_edge + movement_vector[1]
        if laser_x < 0:
            laser_x = cat_right_edge + (offset * movement_vector[0]) + LASER_OFFSET

    cv2.circle(frame, (int(laser_x), int(laser_y)), 5, (255, 255, 255), -1)

def update_prev_centers(current_center):
    """
    Atualiza a lista de centros anteriores do gato.
    """
    prev_centers.append(current_center)
    if len(prev_centers) > FRAME_COUNT_THRESHOLD:
        prev_centers.pop(0)

def detect_cat(frame):
    """
    Detecta o gato no frame usando a rede neural.
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 
        scalefactor=BLOB_SCALE_FACTOR, 
        size=BLOB_SIZE, 
        mean=BLOB_MEAN, 
        swapRB=True, 
        crop=False
    )
    DNN.setInput(blob)
    detections = DNN.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            class_id = int(detections[0, 0, i, 1])
            if class_id == CAT_CLASS_ID:
                x_top_left = int(detections[0, 0, i, 3] * width)
                y_top_left = int(detections[0, 0, i, 4] * height)
                x_bottom_right = int(detections[0, 0, i, 5] * width)
                y_bottom_right = int(detections[0, 0, i, 6] * height)
                bbox = (x_top_left, y_top_left, x_bottom_right - x_top_left, y_bottom_right - y_top_left)
                return bbox
    return None

def track_cat(frame):
    """
    Rastreia o gato no frame atual.
    """
    global tracking, tracker, prev_centers
    success, bbox = tracker.update(frame)
    if success:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        current_center = (x + (w / 2), y + (h / 2))
        cv2.circle(frame, (int(current_center[0]), int(current_center[1])), 5, (0, 0, 255), -1)

        cat_edges = (x, x + w, y + h)

        update_prev_centers(current_center)
        if len(prev_centers) == FRAME_COUNT_THRESHOLD:
            prev_center = prev_centers[0]
            movement_vector = (current_center[0] - prev_center[0], current_center[1] - prev_center[1])
            next_position = (current_center[0] + movement_vector[0], current_center[1] + movement_vector[1])

            direction = "right" if next_position[0] > current_center[0] else "left"
            offset = LASER_MOVEMENT_MULTIPLIER if direction == "right" else -LASER_MOVEMENT_MULTIPLIER
            draw_laser(frame, cat_edges, movement_vector, direction, offset)

            cv2.circle(frame, (int(next_position[0]), int(next_position[1])), 5, (0, 0, 255), -1)
            cv2.line(frame, (int(current_center[0]), int(current_center[1])), (int(next_position[0]), int(next_position[1])), (0, 255, 255), 2)
    else:
        tracking = False
        tracker = cv2.TrackerKCF_create()
        print("Tracking stopped")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not tracking:
        bbox = detect_cat(frame)
        if bbox is not None:
            tracking = True
            tracker.init(frame, bbox)
            x, y, w, h = [int(i) for i in bbox]
            current_center = (x + (w / 2), y + (h / 2))
            cv2.circle(frame, (int(current_center[0]), int(current_center[1])), 5, (0, 0, 255), -1)
            update_prev_centers(current_center)
            print("Tracking started")
    else:
        track_cat(frame)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:
        break

cap.release()
cv2.destroyAllWindows()
