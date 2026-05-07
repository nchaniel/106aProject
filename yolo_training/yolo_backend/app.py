from flask import Flask
from label_studio_ml.api import init_app
from model import YOLOModel

app = init_app(YOLOModel)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)