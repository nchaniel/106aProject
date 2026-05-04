from ultralytics import YOLO
import cv2


class YOLODetector:
    def __init__(self, model_path="best.pt", conf_threshold=0.5):
        """
        model_path: path to YOLO model weights
        conf_threshold: minimum confidence to keep a detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image):
        """
        Runs YOLO on an OpenCV image.

        Args:
            image: OpenCV image (numpy array, BGR format)

        Returns:
            detections: list of dictionaries, each containing:
                - class_id
                - class_name
                - confidence
                - bbox = [x1, y1, x2, y2]
                - center = [cx, cy]
        """
        results = self.model(image, verbose=False)

        detections = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy]
                })

        return detections
    
    def draw_detections(self, image, detections):
        """
        Draws bounding boxes and labels on a copy of the image.
        """
        output = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class_name"]
            conf = det["confidence"]
            cx, cy = det["center"]

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

            label = f"{class_name}: {conf:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output
    
if __name__ == "__main__":
    from pathlib import Path
    import cv2

    current_dir = Path(__file__).resolve().parent
    image_path = current_dir / "test_images" / "test6.jpg"

    print(f"Trying to read image from: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detector = YOLODetector(model_path="best.pt", conf_threshold=0.5)
    detections = detector.detect(image)

    print("Detections:")
    for det in detections:
        print(det)

    vis_image = detector.draw_detections(image, detections)
    display_image = cv2.resize(vis_image, (900, 700))
    cv2.imshow("YOLO Detections", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()