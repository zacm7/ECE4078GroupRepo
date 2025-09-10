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
                box_cord = box.xywh[0]
                box_label = box.cls  # class id
                bounding_boxes.append([
                    prediction.names[int(box_label)],  # class name
                    np.asarray(box_cord),
                    conf
                ])

        # --- Post-processing NMS across all classes ---
        def nms_across_classes(boxes, iou_thresh=0.5):
            # boxes: [label, [x,y,w,h], conf]
            if not boxes:
                return []
            xyxy = np.array([ops.xywh2xyxy(b[1]) for b in boxes])
            confs = np.array([b[2] for b in boxes])
            idxs = cv2.dnn.NMSBoxes(xyxy.tolist(), confs.tolist(), self.conf_thresh, iou_thresh)
            final_boxes = []
            if len(idxs) > 0:
                for i in idxs.flatten():
                    final_boxes.append(boxes[i])
            return final_boxes

        bounding_boxes = nms_across_classes(bounding_boxes, iou_thresh=0.5)
        return bounding_boxes


# FOR TESTING ONLY
if __name__ == '__main__':
    import sys
    import glob
    # Usage: python detector.py <image_folder>
    if len(sys.argv) < 2:
        print("Usage: python detector.py <image_folder>")
        sys.exit(1)
    folder_path = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo = Detector(f'{script_dir}/model/bestv3_new.pt', conf_thresh=0.55)
    # Supported image extensions
    img_exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    img_files = []
    for ext in img_exts:
        img_files.extend(glob.glob(os.path.join(folder_path, ext)))
    if not img_files:
        print(f"No images found in {folder_path}")
        sys.exit(1)
    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue
        bboxes, img_out = yolo.detect_single_image(img)
        print(f"Image: {img_path}")
        print("Detections:", bboxes)
        print("Number of detections:", len(bboxes))
        cv2.imshow('yolo detect', img_out)
        #cv2.waitKey(1000)  # Show each image for 0.5s
        cv2.waitKey(0)