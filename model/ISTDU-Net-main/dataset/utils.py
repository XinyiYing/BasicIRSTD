import numpy as np
import cv2

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32)):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))

    return trans, trans_inv

    # if inv:
    #     trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    # else:
    #     trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    #
    # return trans

PAD = 127
def processGray(im, scale=1., inp_h=None, inp_w=None):
    h,w = im.shape
    new_h = int(h * scale)
    new_w = int(w * scale)
    if inp_w is None:
        inp_w = (new_w | PAD) + 1
    if inp_h is None:
        inp_h = (new_h | PAD) + 1
    c = np.array([new_w // 2,new_h // 2],dtype=np.float32)
    s = np.array([inp_w,inp_h],dtype=np.float32)
    trans_input, trans_input_inv = get_affine_transform(c,s,0,[inp_w,inp_h])
    trans_input = np.float32(trans_input)
    trans_input_inv = np.float32(trans_input_inv)
    resize_im = cv2.resize(im,(new_w,new_h))
    inp_im = cv2.warpAffine(resize_im, trans_input,(inp_w,inp_h),flags=cv2.INTER_LINEAR)
    inp_im = inp_im.reshape(inp_h,inp_w)
    meta = {'c': c, 's': s,
            'in_height': h,  # stride
            'in_width': w,
            'out_height': inp_h,  #stride
            'out_width': inp_w,
            'trans_input': trans_input,
            'trans_input_inv': trans_input_inv}
    return inp_im, meta
