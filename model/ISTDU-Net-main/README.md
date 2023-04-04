# ISTDU-Net
code of article  ISTDU-Net：Infrared Small-Target Detection U-Net


## Introduction
This code using Python3.8+Pytorch1.8 is the complete code of article  ISTDU-Net：Infrared Small-Target Detection U-Net, suitable for Ubuntu system.

## Demo

demo.py：Input picture or video path, frame by frame output algorithm results(the code used to save the results is masked by default, please make your own modifications as required)
The only thing that needs to be changed is the path:
Line10 :  videoPath = './test/images' ##change to the path where you want to test the image

The trained weight path is(already configured in this code)：
./save_pth/ISTDU_Net/best.pth

！！！Note：img, _ = processGray(img, scale=scale, inp_h=inpShape[1], inp_w=inpShape[0])
This function automatically fills images of any size to adapt to the input of the network. Therefore, the larger the input image, the larger the GPU resources will be consumed. 
The input image should preferably be 512 x 512 pixels, a graphics card of 2080Ti is enough.


##The link to download the paper：https://doi.org/10.1109/LGRS.2022.3141584

##Reference：Q. Hou, L. Zhang, F. Tan, Y. Xi, H. Zheng and N. Li, "ISTDU-Net: Infrared Small-Target Detection U-Net," 
in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022, Art no. 7506205, doi: 10.1109/LGRS.2022.3141584.

##Contact us by email：houqingyu@126.com；1762095803@qq.com
