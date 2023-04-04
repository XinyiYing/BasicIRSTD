import cv2
import numpy as np
from detect import detector
from capture import Capture
from dataset import processGray
import time

def main():

    videoPath = './test/images' ##change to the path where you want to test the image

    video = Capture(videoPath, gray=True)
    # inpShape = [256, 256]
    inpShape = [None, None]
    # scale = 0.7
    scale = 1.
    waitKey = 0
    n = 0
    time_all = 0
    # waitKey = 0

    w = int(video.getWeight())
    h = int(video.getHeight())
    # print(w, h)
    d = detector()

    while True:
        ok, img = video.read()
        if not ok:
            break
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (408, 304))
        # img = cv2.resize(img, (512, 512))
        # img = cv2.resize(img, (512, 512))
        # result = img.copy()
        # print(img.shape)
        # cv2.imwrite('./test/img/'+str(n).zfill(6)+'.png',img)

        img, _ = processGray(img, scale=scale, inp_h=inpShape[1], inp_w=inpShape[0])
        # img = cv2.resize(img, (512, 512))
        hp, wp = img.shape
        time_1 = time.time()
        out = d(img)
        time_2 = time.time()
        time_3 = time_2-time_1
        time_all += time_3
        # global time_all
        # print(time_all)

        # print(img.shape)
        # print(out.shape)


        auximg = np.zeros((hp,wp,3), dtype=np.uint8)
        auximg[:, :, 0] = img
        auximg[:, :, 1] = out
        auximg[:, :, 2] = img

        cv2.imshow('out', out)
        cv2.imshow('origin', img)
        cv2.imshow('auximg', auximg)


        # cv2.imwrite('./test/img/'+str(n).zfill(6)+'.png',img)
        # cv2.imwrite('./test/result/'+str(n).zfill(6)+'.png',out*255)
        n+=1

        cv2.waitKey(waitKey)

    # print(time_all)

if __name__ == '__main__':
    main()