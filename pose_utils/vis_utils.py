import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy


def preprocess_image(x, mode='caffe'):
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def input_resize(image, target_size, intrinsics):
    # image: [y, x, c] expected row major
    # target_size: [y, x] expected row major
    # instrinsics: [fx, fy, cx, cy]

    intrinsics = np.asarray(intrinsics)
    y_size, x_size, c_size = image.shape

    if (y_size / x_size) < (target_size[0] / target_size[1]):
        resize_scale = target_size[0] / y_size
        crop = int((x_size - (target_size[1] / resize_scale)) * 0.5)
        image = image[:, crop:(x_size-crop), :]
        image = cv2.resize(image, (int(target_size[1]), int(target_size[0])))
        intrinsics = intrinsics * resize_scale
    else:
        resize_scale = target_size[1] / x_size
        crop = int((y_size - (target_size[0] / resize_scale)) * 0.5)
        image = image[crop:(y_size-crop), :, :]
        image = cv2.resize(image, (int(target_size[1]), int(target_size[0])))
        intrinsics = intrinsics * resize_scale

    return image, intrinsics


def toPix_array(translation, fx, fy, cx, cy):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1)



def draw_3D_bbox_on_image(image, R, t, cam_K, model_info:dict, image_shape=(480,640), factor=0.001, colEst=(0, 205, 205)):

    x_minus = model_info['min_x'] * factor
    y_minus = model_info['min_y'] * factor
    z_minus = model_info['min_z'] * factor
    x_plus = model_info['size_x'] * factor + x_minus
    y_plus = model_info['size_y'] * factor + y_minus
    z_plus = model_info['size_z'] * factor + z_minus

    obj_box = np.array([[x_plus, y_plus, z_plus],
                    [x_plus, y_plus, z_minus],
                    [x_plus, y_minus, z_minus],
                    [x_plus, y_minus, z_plus],
                    [x_minus, y_plus, z_plus],
                    [x_minus, y_plus, z_minus],
                    [x_minus, y_minus, z_minus],
                    [x_minus, y_minus, z_plus]])

    image_raw = copy.deepcopy(image)

    img, intrinsics = input_resize(image,
                                   [image_shape[0], image_shape[1]],
                                   [cam_K[0,0], cam_K[1,1],cam_K[0,2],cam_K[1,2]])  

    ori_points = np.ascontiguousarray(obj_box, dtype=np.float32)
    eDbox = R.dot(ori_points.T).T
    eDbox = eDbox + np.repeat(t[np.newaxis, :], 8, axis=0) # *fac
     # Projection of the bounding box onto the debug image
    eDbox = R.dot(ori_points.T).T
    eDbox = eDbox + np.repeat(t[np.newaxis, :], 8, axis=0)  # * 0.001
    est3D = toPix_array(eDbox, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3])
    eDbox_flat = np.reshape(est3D, (16))
    pose = eDbox_flat.astype(np.uint16)
    pose = np.where(pose < 3, 3, pose)

    image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst, 2)


    return image_raw


def create_debug_image(R_est, t_est, R_gt, t_gt, img, cam_K, model_info, factor, image_shape = (480,640)):
    dbg_img = copy.deepcopy(img)

    dbg_img = draw_3D_bbox_on_image(dbg_img, R_gt, t_gt, cam_K, model_info, factor=factor, image_shape=image_shape, colEst=(255,0,0))

    dbg_img = draw_3D_bbox_on_image(dbg_img, R_est, t_est, cam_K, model_info, factor=factor, image_shape=image_shape, colEst=(0,0,255))

    return dbg_img


