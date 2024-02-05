import argparse
import logging
import os
import time
from threading import Thread

import cv2
import numpy as np
from djitellopy import Tello
from matplotlib import pyplot as plt

import KeyPressModule as kp  # 用于获取键盘按键
from time import sleep

import networks
from evaluate_depth import STEREO_SCALE_FACTOR
from layers import disp_to_depth
from utils import load_weight_file

from paddle.vision import transforms
import paddle.nn.functional as F


def getKeyboardInput(drone, speed, image, flag):
    lr, fb, ud, yv = 0, 0, 0, 0
    key_pressed = 0
    # if kp.getKey("e"):
        # cv2.imwrite('D:/snap-{}.jpg'.format(time.strftime("%H%M%S", time.localtime())), image)
    if kp.getKey("UP"):
        Drone.takeoff()
    elif kp.getKey("DOWN"):
        Drone.land()

    if kp.getKey("j"):
        key_pressed = 1
        Drone.move_left(50)
        lr = -speed
    elif kp.getKey("l"):
        key_pressed = 1
        Drone.move_right(50)
        lr = speed

    if kp.getKey("i"):
        key_pressed = 1
        Drone.move_forward(50)
        fb = speed
    elif kp.getKey("k"):
        key_pressed = 1
        Drone.move_back(50)
        fb = -speed

    if kp.getKey("w"):
        key_pressed = 1
        Drone.move_up(50)
        ud = speed
    elif kp.getKey("s"):
        key_pressed = 1
        Drone.move_down(50)
        ud = -speed

    if kp.getKey("a"):
        key_pressed = 1
        Drone.rotate_counter_clockwise(90)
        yv = -speed
    elif kp.getKey("d"):
        key_pressed = 1
        Drone.rotate_clockwise(90)
        yv = speed
    InfoText = "battery : {0}% height: {1}cm   time: {2}".format(drone.get_battery(), drone.get_height(), time.strftime("%H:%M:%S",time.localtime()))
    # cv2.putText(image, InfoText, (10, 20), font, fontScale, (0, 0, 255), lineThickness)
    if key_pressed == 1:
        InfoText = "Command : lr:{0}% fb:{1} ud:{2} yv:{3}".format(lr, fb, ud, yv)
        # cv2.putText(image, InfoText, (10, 40), font, fontScale, (0, 0, 255), lineThickness)

    # drone.send_rc_control(lr, fb, ud, yv)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')
    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images',
                        default="D:\\Users\\11939\\Desktop\\monodepth2-paddle-main\\assets\\test_data\\")
    parser.add_argument('--load_weights_folder', type=str,
                        help='path to the weight files', default="./fj_best_640x192/weights_best/")
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")

    return parser.parse_args()


# def control():
#     index = 0
#     while True:
#         control_key=["UP","w","i","i","i","i","w","i","DOWN"]
#         key = control_key[index]
#         if key == "UP":
#             Drone.takeoff()
#         elif key == "w":
#             Drone.move_up(90)
#         elif key == "i":
#             Drone.move_forward(50)
#         elif key == "j":
#             Drone.move_left(50)
#         elif key == "DOWN":
#             Drone.land()
#         index += 1
#         # getKeyboardInput(drone=Drone, speed=70, image="", flag=True)  # 按键控制
#         sleep(3)
#         if index == 9:
#             break

def video():
    cnt = 0
    while True:
        OriginalImage = Drone.get_frame_read().frame
        input_image = cv2.resize(OriginalImage, (Camera_Width, Camera_Height))

        origin = input_image.copy()
        original_width, original_height = input_image.shape[1], input_image.shape[0]
        input_image = cv2.resize(input_image, (feed_width, feed_height), interpolation=cv2.INTER_LANCZOS4)
        # input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        t1 = time.time()
        # PREDICTION
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        pred_disp, _ = disp_to_depth(disp, 0.1, 100.0)
        pred_disp = pred_disp.cpu()[:, 0].numpy()
        disp_resized = F.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
        depth = 1.0 / disp_resized
        # depth = STEREO_SCALE_FACTOR * 721 / disp_resized
        # fps = (fps + (1. / (time.time() - t1))) / 2  # 计算平均fps
        # print("151: ", depth.shape)
        # print("153: ", depth2.shape)
        # print(type(depth2))0

        depth2 = np.array(depth[0, 0, :, :])
        # print(type(depth2))

        '''
            避障控制策略
        '''
        leftflag = False
        rightflag = False
        upflag = False
        downflag = False
        # print(depth2.shape)
        depth2 = depth2[0:int(original_height*(2 / 3)), :]
        original_height = int(original_height * (2 / 3))
        # print(depth2.shape)

        threshold = 6
        pixel_threshold = 0
        print(np.min(depth2))

        # plan A
        c = np.sum(depth2 <= threshold)
        d = c > (original_height*original_width)*pixel_threshold
        # if c == 0:
        #     print("Stop Straight")

        # plan B-1
        mid_width = int(original_width / 2)
        mid_height = int(original_height / 2)
        mean_dpeth = np.mean(depth2)
        sum_left = 0
        sum_right = 0
        for i in range(original_height):
            for j in range(mid_width):
                sum_left += depth2[i][j]
        for i in range(original_height):
            for j in range(mid_width, original_width):
                sum_right += depth2[i][j]
        mean_left_depth = sum_left / (original_height*mid_width)
        mean_right_depth = sum_right / ((original_width-mid_width) * original_height)
        if mean_left_depth < mean_right_depth and d:
            print(cnt, " go right")
            rightflag = True
        elif mean_left_depth > mean_right_depth and d:
            print(cnt, " go left")
            leftflag = True
        cnt += 1

        # plan B-2
        mid_width = int(original_width / 2)
        mid_height = int(original_height / 2)
        mean_dpeth = np.mean(depth2)
        sum_up = 0
        sum_bottom = 0
        for i in range(mid_height):
            for j in range(original_width):
                sum_up += depth2[i][j]
        for i in range(mid_height, original_height):
            for j in range(original_width):
                sum_bottom += depth2[i][j]
        mean_up_depth = sum_up / (original_width*mid_height)
        mean_bottom_depth = sum_bottom / ((original_height-mid_height) * original_width)
        if mean_up_depth < mean_bottom_depth and d:
            print(cnt, " go down")
            downflag = True
        elif mean_up_depth > mean_bottom_depth and d:
            print(cnt, " go up")
            upflag = True
        cnt += 1

        if (rightflag and downflag) or (rightflag and upflag):
            downflag = False
            upflag = False
        elif (leftflag and downflag) or (leftflag and upflag):
            downflag = False
            upflag = False

        # Saving numpy file
        # output_directory = os.path.dirname(args.image_path)
        # output_name = os.path.splitext(os.path.basename(args.image_path))[0]
        # output_directory = args.image_path
        # output_name = os.path.splitext(os.path.basename(image_path))[0]
        # scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
        # np.save(name_dest_npy, scaled_disp.cpu().numpy())'

        disp_resized = disp_resized.cpu()[0].numpy()
        disp_resized = np.transpose(disp_resized, (1, 2, 0))
        rescaled = disp_resized[:, :, 0]
        rescaled = rescaled - np.min(rescaled)
        rescaled = rescaled / np.max(rescaled)
        plasma = plt.get_cmap('magma')
        pred = plasma(rescaled)[:, :, :3]
        pred = np.array(pred * 255.0, dtype=np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        viz = np.concatenate((origin, pred), axis=1)

        cv2.imshow("Drone Control Centre", pred)
        cv2.waitKey(1)
        # video_out.write(origin)


if __name__ == '__main__':

    # 主程序
    # 摄像头设置
    Camera_Width = 720
    Camera_Height = 480
    DetectRange = [6000, 11000]  # DetectRange[0] 是保持静止的检测人脸面积阈值下限，DetectRange[0] 是保持静止的检测人脸面积阈值上限
    PID_Parameter = [0.5, 0.0004, 0.4]
    pErrorRotate, pErrorUp = 0, 0

    # 字体设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (255, 0, 0)
    lineThickness = 1

    # Tello初始化设置
    Drone = Tello()  # 创建飞行器对象
    Drone.connect()  # 连接到飞行器
    Drone.streamon()  # 开启视频传输
    Drone.LOGGER.setLevel(logging.ERROR)  # 只显示错误信息
    sleep(5)  #  等待视频初始化
    kp.init()  # 初始化按键处理模块

    args = parse_args()
    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder")
    depth_decoder_path = os.path.join(args.load_weights_folder, "depth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, 'scratch', use_aspp=True)
    loaded_dict_enc = load_weight_file(encoder_path)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_dict(filtered_dict_enc)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, att_type='CA', use_frelu=True)
    loaded_dict = load_weight_file(depth_decoder_path)
    depth_decoder.load_dict(loaded_dict)
    depth_decoder.eval()
    fps = 0.0  # 计算帧数
    cnt = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    myfps = 50
    # video_out = cv2.VideoWriter(str(time.time())+'.avi', fourcc, myfps, (Camera_Width, Camera_Height))

    # t1 = Thread(target=control)
    # t1.start()
    # t2 = Thread(target=video)
    # t2.start()

    video()