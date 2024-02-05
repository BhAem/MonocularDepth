# _*_ coding:utf-8 _*_

import cv2
import os

def video_get_img(videoPath, svPath):
    #读取视频
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    fps = cap.get(cv2.CAP_PROP_FPS) #获取视频的帧率
    print(fps)
    while True:
        if cap.grab():
            '''
            flag:按帧读取视频，返回值ret是布尔型，正确读取则返回True
            frame:为每一帧的图像
            '''
            flag, frame = cap.retrieve()  #解码,并返回捕获的视频帧
            if not flag:
                continue
            else:
                cv2.imshow('video', frame)
                numFrame += 1
                #拼接图片保存路径
                newPath = svPath + "\\img" + str(numFrame) + ".jpg"
                #将图片按照设置格式，保存到文件
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
                # if numFrame == 10:
                #     #只保存10张图片
                #     cap.release() #释放读取画面状态
                #     break         #结束循环
                # else:
                #     pass
        #waitKey()函数的功能是不断刷新图像,用于显示图像的作用，频率时间为delay , 单位为ms
        if cv2.waitKey(10) == 27: #用户按下ESC(ASCII码为27) ,窗口将会退出
            break


if __name__ == '__main__':
    #视频路径
    videopath = r'D:\Users\11939\Desktop\monodepth2-paddle-main\2.mp4'
    #图片保存路径
    svpath   = r'D:\Users\11939\Desktop\monodepth2-paddle-main\video2pic'
    if os.path.exists(svpath):
        pass
    else:
        os.mkdir(svpath)
    video_get_img(videopath, svpath)
