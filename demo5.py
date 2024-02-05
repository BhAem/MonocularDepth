import os
import sys
import glob
import argparse
import time

import cv2
import matplotlib.pyplot
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import paddle
import paddle.nn.functional as F
from matplotlib import pyplot as plt
from paddle.vision import transforms

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR
from utils import load_weight_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images',
                        default="D:\\Users\\11939\\Desktop\\monodepth2-paddle-main\\video2pic\\")
    parser.add_argument('--video_path', type=str,
                        help='path to a test image or folder of images',
                        default="002.avi")
    parser.add_argument('--output_path', type=str,
                        help='path to a test image or folder of images',
                        default="D:\\Users\\11939\\Desktop\\monodepth2-paddle-main\\video2pic_output\\")
    parser.add_argument('--load_weights_folder', type=str,
                        help='path to the weight files', default="./fj_best_640x192/weights_best/")
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")

    return parser.parse_args()


def test_simple(args):
    """
    Function to predict for a single image or folder of images
    """

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



    cap = cv2.VideoCapture(args.video_path)
    # 读取一帧
    ret, frame = cap.read()
    # 视频写入格式
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 视频输出对象
    out = cv2.VideoWriter('002.mp4', fourcc, fps, size)



    # pic_lists = os.listdir(args.image_path)
    # pic_dirs = []
    # for pic_list in pic_lists:
    #     pic_dir = os.path.join(args.image_path, pic_list)
    #     pic_dirs.append(pic_dir)

    fps = 0.0  # 计算帧数
    # PREDICTING ON EACH IMAGE IN TURN
    with paddle.no_grad():

        while ret:
            ret, input_image = cap.read()
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            orignal = input_image.copy()
            if input_image is None:
                print('read frame is err!')
                continue
        # for image_path in pic_dirs:
            # Load image and preprocess
            # input_image = pil.open(args.image_path).convert('RGB')
            # input_image = pil.open(image_path).convert('RGB')
            original_height, original_width, _ = input_image.shape
            input_image = cv2.resize(input_image, (feed_width, feed_height), cv2.INTER_LANCZOS4)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            t1 = time.time()
            # PREDICTION
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = F.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
            # print("92: ", disp_resized.shape)
            fps = (fps + (1. / (time.time() - t1))) / 2  # 计算平均fps
            # Saving numpy file
            # output_directory = os.path.dirname(args.image_path)
            # output_name = os.path.splitext(os.path.basename(args.image_path))[0]
            # output_directory = args.output_path
            # output_name = os.path.splitext(os.path.basename(image_path))[0]
            # scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # # Saving colormapped depth image
            # disp_resized2 = disp_resized.cpu()[0].numpy()
            # # print("105: ", disp_resized2.shape)
            # disp_resized_np = disp_resized.squeeze().cpu().numpy()
            # # print("107: ", disp_resized_np.shape)
            # vmax = np.percentile(disp_resized_np, 95)
            # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            # # print("112: ", colormapped_im.shape)
            # im = pil.fromarray(colormapped_im)
            #
            # name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            # im.save(name_dest_im)

            disp_resized = disp_resized.cpu()[0].numpy()
            disp_resized = np.transpose(disp_resized, (1, 2, 0))
            rescaled = disp_resized[:, :, 0]
            rescaled = rescaled - np.min(rescaled)
            rescaled = rescaled / np.max(rescaled)
            plasma = plt.get_cmap('magma')
            pred = plasma(rescaled)[:, :, :3]
            pred = np.array(pred * 255.0, dtype=np.uint8)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            orignal = cv2.cvtColor(orignal, cv2.COLOR_BGR2RGB)
            viz = np.concatenate((orignal, pred), axis=1)
            cv2.imshow("Drone Control Centre", viz)
            cv2.waitKey(1)
            out.write(viz)

    # 释放窗口
    cv2.destroyAllWindows()
    # 视频写入结束
    cap.release()

    print(fps)
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)