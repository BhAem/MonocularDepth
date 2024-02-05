import cv2
import os

img_root = 'video2pic_output/' #照片文件路径
fps = 60    #FPS
size=(640, 352)    #图片、视频尺寸
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter('./demofps60.mp4',fourcc,fps,size, True)

for img_file in os.listdir(img_root):
    frame = cv2.imread(img_root + img_file)
    videoWriter.write(frame)
    print(img_root + img_file + ' done!')
videoWriter.release()
