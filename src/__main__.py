import cv2

CAT_CLASS_ID = 8
PROTO_TXT = "model/MobileNetSSD_deploy.prototxt"
CAFFE_MODEL = "model/MobileNetSSD_deploy.caffemodel"

DNN = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)

cap = cv2.VideoCapture("data/3.mp4")
tracker = cv2.TrackerKCF_create()
tracking = False

prev_centers = []
frame_count = 4

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
                x, y, w, h = [int(i) for i in bbox]
                current_center = (x + (w/2), y + (h/2))
                cv2.circle(frame, (int(current_center[0]), int(current_center[1])), 5, (0, 0, 255), -1)

                cat_left_edge = x
                cat_right_edge = x + w

                height, width, _ = frame.shape

                right_edge = width-1
                left_edge = 0

                prev_centers.append(current_center)
            
                if len(prev_centers) > frame_count:
                    prev_centers = prev_centers[-frame_count:]

                if len(prev_centers) == frame_count:
                    prev_center = prev_centers[0]            
                    movement_vector = (current_center[0] - prev_center[0], current_center[1] - prev_center[1])
                    next_position = (current_center[0] + movement_vector[0], current_center[1] + movement_vector[1])

                    if next_position[0] > current_center[0]:
                        laser_offset_x = 2

                        laser_x = cat_right_edge + (laser_offset_x * movement_vector[0]) + 50
                        laser_y = cat_bottom_edge + movement_vector[1]
                        cv2.circle(frame, (int(laser_x), int(laser_y)), 5, (255, 255, 255), -1)
                    else: 
                        laser_offset_x = -2

                        laser_x = cat_left_edge - (laser_offset_x * movement_vector[0]) - 50
                        laser_y = cat_bottom_edge + movement_vector[1]
                        cv2.circle(frame, (int(laser_x), int(laser_y)), 5, (255, 255, 255), -1)
                        
                print("Tracking started")

    else:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            current_center = (x + (w/2), y + (h/2))
            cv2.circle(frame, (int(current_center[0]), int(current_center[1])), 5, (0, 0, 255), -1)

            cat_left_edge = x
            cat_right_edge = x + w
            cat_bottom_edge = y + h

            height, width, _ = frame.shape

            right_edge = width-1
            left_edge = 0

            prev_centers.append(current_center)
        
            if len(prev_centers) > frame_count:
               prev_centers = prev_centers[-frame_count:]
        
            if len(prev_centers) == frame_count:
                prev_center = prev_centers[0]            
                movement_vector = (current_center[0] - prev_center[0], current_center[1] - prev_center[1])
                next_position = (current_center[0] + movement_vector[0], current_center[1] + movement_vector[1])

                if next_position[0] > current_center[0]:
                    laser_offset_x = 2

                    laser_x = cat_right_edge + (laser_offset_x * movement_vector[0]) + 50
                    laser_y = cat_bottom_edge + movement_vector[1]
                    cv2.circle(frame, (int(laser_x), int(laser_y)), 5, (255, 255, 255), -1)
                else: 
                    laser_offset_x = -2

                    laser_x = cat_left_edge - (laser_offset_x * movement_vector[0]) - 50
                    laser_y = cat_bottom_edge + movement_vector[1]
                    cv2.circle(frame, (int(laser_x), int(laser_y)), 5, (255, 255, 255), -1)

                cv2.circle(frame, (int(next_position[0]), int(next_position[1])), 5, (0, 0, 255), -1)
                cv2.line(frame, (int(current_center[0]), int(current_center[1])), (int(next_position[0]), int(next_position[1])), (0, 255, 255), 2)
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
