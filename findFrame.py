import numpy as np
import torch
import torchvision


def predict_keypts(image):
    global model

    image = image[:, :, ::-1]
    img_tensor = torchvision.transforms.functional.to_tensor(image.copy()).cuda()
    output = model([img_tensor])[0]

    keypts = output['keypoints'].cpu().numpy()
    if len(keypts) > 0:
        keypts = keypts[0]
        keypts = keypts[:, :-1]
    else:
        keypts = None
    return keypts


def select_joints(keypoints):
    ret_pts = []
    selected_joints = ['left_shoulder',
                       'right_shoulder',
                       'left_elbow',
                       'right_elbow',
                       'left_wrist',
                       'right_wrist',
                       'left_hip',
                       'right_hip']

    joints_index = {'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
                    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
                    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
                    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16}

    location = keypoints.astype(np.uint16).reshape(-1)
    for joint in selected_joints:
        assert joint in joints_index, 'JOINT' + joint + ' DONT EXIST!'
        ret_pts.append(location[2 * joints_index[joint]])
        ret_pts.append(location[2 * joints_index[joint] + 1])

    return ret_pts


def isStart(joints_loc):
    action = False

    shouder_lr = joints_loc[0:4]  # 左肩坐标(x1,y1) 右肩坐标(x2,y2)
    elbow_lr = joints_loc[4:8]
    wrist_lr = joints_loc[8:12]
    hip_lr = joints_loc[12:16]

    shouder_y = (shouder_lr[1] + shouder_lr[3]) // 2
    hip_y = (hip_lr[1] + hip_lr[3]) // 2
    threshold_y = shouder_y + 3 * (hip_y - shouder_y) // 4  # 水平阈值线

    ''' 比较 手腕抬起高度 '''
    if wrist_lr[1] < threshold_y or wrist_lr[3] < threshold_y:
        action = True

    return action


def isEnd(joints_loc):
    action = False

    shouder_lr = joints_loc[0:4]  # 左肩坐标(x1,y1) 右肩坐标(x2,y2)
    elbow_lr = joints_loc[4:8]
    wrist_lr = joints_loc[8:12]
    hip_lr = joints_loc[12:16]

    shouder_y = (shouder_lr[1] + shouder_lr[3]) // 2
    hip_y = (hip_lr[1] + hip_lr[3]) // 2
    threshold_y = shouder_y + 6 * (hip_y - shouder_y) // 7  # 水平阈值线

    ''' 比较 手腕放下高度 '''
    if wrist_lr[1] > threshold_y and wrist_lr[3] > threshold_y:
        action = True

    return action


torch.set_grad_enabled(False)
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model = model.eval().cuda()

# if __name__ == '__main__':
#
#     cap = cv2.VideoCapture('手语识别_1/004.avi')  # 读视频文件
#     index = 0
#     find = False
#
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
#         index += 1
#
#         if index >= 36 and not find:
#             keypts = predict_keypts(frame)
#             if keypts is not None:
#                 target_loc = select_joints(keypts)
#                 status = isEnd(target_loc)
#
#                 if status is True:
#                     print('End Index is {:}'.format(index))
#                     find = True
#
#                     # for demo show
#                     frame = cv2.resize(frame, (640, 360))
#                     frame = cv2.putText(frame, "The End", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2,
#                                         cv2.LINE_8)
#                     cv2.imshow('frame', frame)
#                     cv2.waitKey(0)
#
#         else:
#             # for imshow
#             frame = cv2.resize(frame, (640, 360))
#             cv2.imshow('frame', frame)
#         if cv2.waitKey(5) & 0xFF == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#     cap.release()
