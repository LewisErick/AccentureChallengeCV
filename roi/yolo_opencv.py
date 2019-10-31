import cv2
import numpy as np

def get_objects(img, config="yolov3-tiny.cfg",
                weights="yolov3-tiny.weights",
                classes_="yolov3.txt"):
   
    def get_output_layers(net):
        
        layer_names = net.getLayerNames()
        
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers


    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        
    image = cv2.imread(img)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(classes_, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights, config)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    objects = []

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        objects.append(
            (str(classes[class_ids[i]]),
            image[round(y):round(y+h), round(x):round(x+w)])
        )

    return objects


def get_output_layers(net):
        
        layer_names = net.getLayerNames()
        
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_objects_video(vid, config="yolov3-tiny.cfg",
                weights="yolov3-tiny.weights",
                classes_="yolov3.txt",
                sample_rate=1):

    cam = cv2.VideoCapture(vid)

    frames = []

    frame_index = 0
    
    while(True): 
        ret = None
        frame = None
        # reading from frame 
        if sample_rate > 1:
            for _ in range(sample_rate):
                ret,frame = cam.read()
    
        if ret:
            image = frame
            Width = image.shape[1]
            Height = image.shape[0]
            scale = 0.00392

            classes = None

            with open(classes_, 'r') as f:
                classes = [line.strip() for line in f.readlines()]

            COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

            net = cv2.dnn.readNet(weights, config)

            blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

            net.setInput(blob)

            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4


            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])


            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            objects = []

            for i in indices:
                i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                objects.append(
                    (str(classes[class_ids[i]]),
                    image[round(y):round(y+h), round(x):round(x+w)])
                )
        else:
            break
        frame_index = frame_index + 1
        frames.append(objects)
    return frames
