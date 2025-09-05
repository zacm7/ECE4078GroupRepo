import cv2
import os
import numpy as np
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils import ops


class Detector:
    def __init__(self, model_path, conf_thresh=0.55):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh  # confidence threshold

        self.class_colour = {
            'orange': (0, 165, 255),
            'lemon': (0, 255, 255),
            'pear': (0, 255, 0),
            'tomato': (0, 0, 255),
            'capsicum': (255, 0, 0),
            'potato': (255, 255, 0),
            'pumpkin': (255, 165, 0),
            'garlic': (255, 0, 255)
        }

    def detect_single_image(self, img):
        """
        Detect target(s) in an image and return bounding boxes + annotated image
        """
        bboxes = self._get_bounding_boxes(img)

        img_out = deepcopy(img)

        # draw bounding boxes on the image
        for bbox in bboxes:
            # unpack
            label = bbox[0]
            box_xywh = bbox[1]
            conf = bbox[2]

            # convert [x,y,w,h] to [x1,y1,x2,y2]
            xyxy = ops.xywh2xyxy(box_xywh)
            x1, y1, x2, y2 = map(int, xyxy)

            # draw bounding box
            img_out = cv2.rectangle(
                img_out, (x1, y1), (x2, y2),
                self.class_colour[label.lower()],
                thickness=2
            )

            # draw class label + confidence
            text = f"{label} {conf:.2f}"
            img_out = cv2.putText(
                img_out, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                self.class_colour[label.lower()], 2
            )

        return bboxes, img_out

    def _get_bounding_boxes(self, cv_img):
        """
        Run YOLO prediction and return bounding boxes
        Format: [label, [x,y,w,h], confidence]
        """
        predictions = self.model.predict(cv_img, imgsz=320, verbose=False)

        bounding_boxes = []
        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.conf_thresh:  # skip low confidence detections
                    continue

                # bounding format [x, y, w, h]
                box_cord = box.xywh[0]
                box_label = box.cls  # class id

                bounding_boxes.append([
                    prediction.names[int(box_label)],  # class name
                    np.asarray(box_cord),
                    conf
                ])

        return bounding_boxes


# FOR TESTING ONLY
if __name__ == '__main__':
    # get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    yolo = Detector(f'{script_dir}/model/bestv2.pt', conf_thresh=0.55)

    img = cv2.imread(f'{script_dir}/test/img_81.png')

    bboxes, img_out = yolo.detect_single_image(img)

    print("Detections:", bboxes)
    print("Number of detections:", len(bboxes))

    cv2.imshow('yolo detect', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
