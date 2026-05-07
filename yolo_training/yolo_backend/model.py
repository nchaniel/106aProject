from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
import urllib.parse
import os

# for making labeling faster by using the existing yolo model to predict bounding boxes

class YOLOModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "updated.pt"
        )

        self.model = YOLO(model_path)

    def ls_path_to_file(self, path: str):

        if not isinstance(path, str):
            return path

        if "local-files" not in path:
            return path

        parsed = urllib.parse.urlparse(path)
        query = urllib.parse.parse_qs(parsed.query)

        rel_path = query.get("d", [None])[0]

        if rel_path is None:
            return path

        rel_path = urllib.parse.unquote(rel_path)

        ROOT = r"C:\Users\natha\Documents\MEC106A"

        return os.path.join(ROOT, rel_path)

    def predict(self, tasks, **kwargs):

        results = []

        for task in tasks:

            image_url = task["data"]["image"]
            image_path = self.ls_path_to_file(image_url)

            yolo_results = self.model.predict(
                source=image_path,
                verbose=False
            )

            output = []

            for r in yolo_results:

                if r.boxes is None:
                    continue

                img_width = r.orig_shape[1]
                img_height = r.orig_shape[0]

                for b in r.boxes:

                    x1, y1, x2, y2 = b.xyxy[0].tolist()

                    x = (x1 / img_width) * 100
                    y = (y1 / img_height) * 100
                    w = ((x2 - x1) / img_width) * 100
                    h = ((y2 - y1) / img_height) * 100

                    conf = float(b.conf[0])

                    if conf < 0.4:
                        continue

                    cls = int(b.cls[0])

                    label_name = self.model.names[cls]

                    output.append({
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "rectanglelabels": [label_name]
                        },
                        "score": conf
                    })

            results.append({"result": output})

        return results