import cv2

CAT_CLASS_ID = 8
PROTO_TXT = "model/MobileNetSSD_deploy.prototxt"
CAFFE_MODEL = "model/MobileNetSSD_deploy.caffemodel"

DNN = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)

cap = cv2.VideoCapture("data/4.mp4")
tracker = cv2.TrackerKCF_create()
tracking = False

while True:
    ret, frame = cap.read()

    if not tracking:
        height, width = frame.shape[:2]

        BLOB = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 127.5,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        DNN.setInput(BLOB)
        detections = DNN.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                class_id = int(detections[0, 0, i, 1])

                if class_id != CAT_CLASS_ID:
                    continue

                x_top_left = int(detections[0, 0, i, 3] * width)
                y_top_left = int(detections[0, 0, i, 4] * height)
                x_bottom_right = int(detections[0, 0, i, 5] * width)
                y_bottom_right = int(detections[0, 0, i, 6] * height)

                bbox = (x_top_left, y_top_left, x_bottom_right - x_top_left, y_bottom_right - y_top_left)

                tracking = True
                tracker.init(frame, bbox)
                print("Tracking started")

    else:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            tracking = False
            tracker = cv2.TrackerKCF_create()
            print("Tracking stopped")

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:
        break

cap.release()
cv2.destroyAllWindows()
