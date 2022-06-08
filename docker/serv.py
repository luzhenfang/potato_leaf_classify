import io
import json

import numpy as np
from PIL import Image
from flask import Flask, request, make_response, send_file, render_template
import torch
from flask_cors import CORS

from net import ResNet, index_to_label
from torchvision import transforms
from gevent import pywsgi

Settings = {
    "port": 5000,
    "host": "0.0.0.0",
    "debug": False
}

DEBUG_MODE = Settings["debug"]

MODEL_PATH = "/data/model.pth" if not DEBUG_MODE else "../models/linear_model_27__99.2.pth"
CRT_PATH = "/data/server.crt" if not DEBUG_MODE else ""
KEY_PATH = "/data/server.key" if not DEBUG_MODE else ""

app = Flask(__name__)
# 开启跨域
CORS(app, supports_credentials=True)


class CNN:
    def __init__(self, model_path):
        self.net = torch.load(model_path, map_location=torch.device("cpu"))
        self.net.eval()

    def trans_img(self, img):
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.5, 0.5, 0.5))
        ])
        return trans(img)

    # 模型预测
    def predict(self, x):
        with torch.no_grad():
            output = self.net(x)
            conf, pre = torch.max(torch.softmax(output, dim=1), dim=1)
            return str(conf.item())[:8], pre.item()


# 创建预测模型
cnn = CNN(MODEL_PATH)


@app.route("/test")
def test():
    return "Server,OK!"


@app.route("/")
def report():
    return render_template("report.html")


# 预测接口
@app.route("/predict", methods=["post"])
def predict():
    try:
        # file = request.files.get("file")
        # img = Image.open(file.stream)
        # img = cnn.trans_img(img)
        # conf, val = cnn.predict(torch.reshape(img, (1, 3, 224, 224)))

        blob = request.files['file'].read()
        image = Image.open(io.BytesIO(blob))
        in_tensor = cnn.trans_img(image)
        conf, val = cnn.predict(torch.reshape(in_tensor, (1, 3, 224, 224)))
        resp = {
            "code": 200,
            "msg": "识别成功!",
            "type": index_to_label[val],
            "conf": conf
        }
        return json.dumps(resp, ensure_ascii=False)
    except Exception as e:
        print(e)
        return json.dumps({
            "code": -1,
            "msg": "服务器出现异常!"
        })


if __name__ == '__main__':
    if DEBUG_MODE:
        print("==========你看到现在这条消息说明当前环境为调试模式===========", "如果当前环境为生产环境中请谨慎使用", sep='\n')
        app.run(host="0.0.0.0", port=5001, processes=True)
    else:
        server = pywsgi.WSGIServer((Settings["host"], Settings["port"]), app, keyfile=KEY_PATH, certfile=CRT_PATH)
        server.serve_forever()
