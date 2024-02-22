import cv2
import numpy as np
import copy
import torch
import os
from extractor import ViTExtractor
import json
import matplotlib.pyplot as plt
from src.inspect_similarity import chunk_cosine_sim
import sys
from extractor import ViTExtractor
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from itertools import combinations
from tqdm import tqdm
import torch.nn.functional as F

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


def extract_describtor(img_path, extractor, device):    
    patch_size = extractor.model.patch_embed.patch_size
    image_batch_a, image_pil_a = extractor.preprocess(f"{img_path}", 
                                                      load_size=224)
    #image_batch_b, image_pil_b = extractor.preprocess(image_path_b, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer=11, facet='key', bin = False, include_cls=True)
    #descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer=9, facet='key', bin = False, include_cls=False)
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size
    
    return descs_a

def make_quadratic_crop(image, bbox):
    # Define the bounding box
    x_left, y_top, width, height = bbox

    # Calculate the size of the square crop based on the longer side
    longer_side = max(width, height)
    crop_size = (longer_side, longer_side)

    # Calculate the center of the bounding box
    center_x = x_left + width / 2
    center_y = y_top + height / 2
    crop_size = min(longer_side, int(max(width/2, height/2) * 2))

    # Calculate the coordinates of the top-left corner of the square crop
    crop_x = int(center_x - crop_size / 2)
    crop_y = int(center_y - crop_size / 2)

    # Check if the crop goes beyond the image boundaries
    if crop_x < 0 or crop_y < 0 or crop_x + crop_size > image.shape[1] or crop_y + crop_size > image.shape[0]:

        # If the crop goes beyond the image boundaries, crop first and add a border using cv2.copyMakeBorder to make the crop quadratic
        crop = image[max(crop_y, 0):min(crop_y+crop_size, image.shape[0]), max(crop_x, 0):min(crop_x+crop_size, image.shape[1])]
        border_size = max(crop_size - crop.shape[1], crop_size - crop.shape[0])
        border_size = max(0, border_size)  # Make sure the border size is not negative
        
        
        if crop_x < 0 or crop_x + crop_size > image.shape[1]:
            left = border_size // 2
            right = border_size - left
            crop = cv2.copyMakeBorder(crop, 0, 0, left, right, cv2.BORDER_REPLICATE)
        elif crop_y < 0 or crop_y + crop_size > image.shape[0]:
            top = border_size // 2
            bottom = border_size - top
            crop = cv2.copyMakeBorder(crop, top, bottom, 0, 0, cv2.BORDER_REPLICATE)
        else:
            print("Something went wrong during rectifying crop")
            return None

    else:
        # If the crop is within the image boundaries, just crop the image
        crop = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    
    return crop


def find_template(desc_input, desc_templates, obj_id, num_results):

    similarities = [(1-cosine(desc_input, desc_template), i) for i, desc_template in enumerate(desc_templates) 
                    if desc_input.shape==desc_template.shape]
    
    sorted_sims = sorted(similarities, key=lambda x:x[0], reverse=True)
    
    return [(sim[0],sim[1],obj_id) for sim in sorted_sims[:num_results]]


def calculate_score(pred_location, gt_location, id_symmetry, id_obj, pred_id_obj):
    unique_ids, inverse_indices = torch.unique(id_obj, sorted=True, return_inverse=True)
    cosine_sim = F.cosine_similarity(pred_location, gt_location)
    angle_err = torch.rad2deg(torch.arccos(cosine_sim.clamp(min=-1, max=1)))

    # for symmetry
    gt_location_opposite = gt_location
    gt_location_opposite[:, :2] *= -1  # rotation 180 in Z axis
    cosine_sim_sym = F.cosine_similarity(gt_location_opposite, gt_location_opposite)
    angle_err_sym = torch.rad2deg(torch.arccos(cosine_sim_sym.clamp(min=-1, max=1)))
    angle_err[id_symmetry == 1] = torch.minimum(angle_err[id_symmetry == 1], angle_err_sym[id_symmetry == 1])

    list_err, list_pose_acc, list_class_acc, list_class_and_pose_acc15 = {}, {}, {}, {}
    for i in range(len(unique_ids)):
        err = angle_err[id_obj == unique_ids[i]]
        recognition_acc = (pred_id_obj[id_obj == unique_ids[i]] == unique_ids[i])
        
        class_and_pose_acc15 = torch.logical_and(err <= 15, recognition_acc).float().mean()
        err = err.mean()
        recognition_acc = recognition_acc.float().mean()
        pose_acc = (err <= 15).float().mean()

        list_err[unique_ids[i].item()] = err
        list_pose_acc[unique_ids[i].item()] = pose_acc
        list_class_acc[unique_ids[i].item()] = recognition_acc
        list_class_and_pose_acc15[unique_ids[i].item()] = class_and_pose_acc15

    list_err["mean"] = torch.mean(angle_err)
    list_pose_acc["mean"] = (angle_err <= 15).float().mean()
    list_class_acc["mean"] = (pred_id_obj == id_obj).float().mean()
    list_class_and_pose_acc15["mean"] = torch.logical_and(angle_err <= 15, pred_id_obj == id_obj).float().mean()

    return list_err, list_pose_acc, list_class_acc, list_class_and_pose_acc15
    
                 

