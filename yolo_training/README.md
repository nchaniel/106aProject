Workflow (it needs improvement, I know)
Image Labeling:
    Terminal Commands:
        Activate label_env: .\label_env\Scripts\Activate.ps1
        Allow access to local storage: $env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"    
        Allow access to local storage: $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="C:\PATH TO IMAGES"
        Start Label Studio: label-studio start
    Label Studio UI: 
        Connect to local storage with images, connect ML model with YOLO for predictions, create/fix annotations, export YOLO txt files

Yolo Training:
    Pre-processing:
        Add exported labels to "broken_labels"
        Add corresponding images to "jpg_images_new" (path can be adjusted for the next step)
        Run repair_dataset.py
        Move images and labels to respective "train" file in "dataset"
        Run split_dataset.py
    Terminal Commands:
        Activate yolo_env: .\yolo_env\Scripts\Activate.ps1
        Train Model: yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 (default, change model to path to actual model)
        Test Model: yolo detect predict model=runs/detect/train/weights/best.pt source="image name"

ML Model to provide bounding box suggestions:
    Activate ls_ml_env: .\ls_ml_env\Scripts\activate
    Start model: python yolo_backend\model.py
    Take URL, connect to Label Studio in its settings.
