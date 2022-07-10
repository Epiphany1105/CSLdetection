import cv2
import torch
import torchvision
from torch import optim
import numpy as np
from RNN import RNN


def transform_keypts(x_pts, y_pts):
    x_pts = np.array(x_pts, dtype=np.float32)
    y_pts = np.array(y_pts, dtype=np.float32)
    # 归一化指标
    c1 = np.mean(x_pts)
    c2 = np.mean(y_pts)
    s1 = np.std(x_pts)
    s2 = np.std(y_pts)
    # 位置归一化
    x_pts -= c1
    y_pts -= c2
    # 尺度归一化
    x_pts /= s1
    y_pts /= s2

    return list(x_pts), list(y_pts)


def predict_keypts(image):
    image = image[:, :, ::-1]
    img_tensor = torchvision.transforms.functional.to_tensor(image.copy()).to(device)
    output = model_keypts([img_tensor])[0]

    # scores = output['scores'].cpu().numpy()
    scores = output['scores'].cpu().detach().numpy()
    top = np.argmax(scores)
    keypts = output['keypoints'].cpu().detach().numpy()[top]
    keypts = keypts[:, :-1]
    location = keypts.astype(np.uint16).reshape(-1)

    return location[0::2], location[1::2]


def load_model(filepath, input_size=22, hidden_size=256, num_classes=500):
    model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)
    return model


def label2text(key, dict_path='./dictionary.txt'):
    txts = {}
    with open(dict_path, 'r', encoding='UTF-8') as label_dict:
        for i, line in enumerate(label_dict.readlines()):
            line = line.strip()
            words = line.split('\t')
            txts[i] = words[1]

    return txts[int(key)]


torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_keypts = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model_keypts = model_keypts.eval().to(device)

# model = load_model('best1.mdl')
# model.to(device)

LineThick = 3
PointThick = 4


def OpenPoseDetect(frame):
    x_loc, y_loc = predict_keypts(frame)
    for i in range(15):
        img = cv2.circle(frame, (x_loc[i], y_loc[i]), PointThick, (0, 0, 255), -1)
    img = cv2.line(img, (x_loc[0], y_loc[0]), (x_loc[1], y_loc[1]), (193, 182, 255), LineThick)
    img = cv2.line(img, (x_loc[0], y_loc[0]), (x_loc[2], y_loc[2]), (193, 182, 255), LineThick)
    img = cv2.line(img, (x_loc[1], y_loc[1]), (x_loc[3], y_loc[3]), (193, 182, 255), LineThick)
    img = cv2.line(img, (x_loc[2], y_loc[2]), (x_loc[4], y_loc[4]), (193, 182, 255), LineThick)
    img = cv2.line(img, (x_loc[0], y_loc[0]), (x_loc[11], y_loc[11]), (16, 144, 246), LineThick)
    img = cv2.line(img, (x_loc[0], y_loc[0]), (x_loc[12], y_loc[12]), (16, 144, 246), LineThick)
    img = cv2.line(img, (x_loc[6], y_loc[6]), (x_loc[8], y_loc[8]), (16, 144, 246), LineThick)
    img = cv2.line(img, (x_loc[8], y_loc[8]), (x_loc[10], y_loc[10]), (1, 240, 255), LineThick)
    img = cv2.line(img, (x_loc[5], y_loc[5]), (x_loc[7], y_loc[7]), (1, 240, 255), LineThick)
    img = cv2.line(img, (x_loc[7], y_loc[7]), (x_loc[9], y_loc[9]), (140, 47, 240), LineThick)
    img = cv2.line(img, (x_loc[11], y_loc[11]), (x_loc[13], y_loc[13]), (140, 47, 240), LineThick)
    img = cv2.line(img, (x_loc[12], y_loc[12]), (x_loc[14], y_loc[14]), (140, 47, 240), LineThick)
    return img
