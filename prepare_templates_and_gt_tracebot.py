from rendering.renderer_xyz import Renderer
from rendering.model import Model3D
from rendering import utils as renderutil
import os
import json
import cv2
import numpy as np
import transforms3d as tf3d
from pose_utils import vis_utils
from pose_utils import img_utils
import matplotlib.pyplot as plt
import argparse
from pose_extractor import PoseViTExtractor
# from pose_extractor_dinov2 import PoseViTExtractor
from tqdm import tqdm
from PIL import Image
import torch

def get_bbox_and_segmentation(depth_path):
    # Load the depth image
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    # Create a binary segmentation mask by checking if the depth value is greater than 0
    segmentation_mask = np.where(depth_image != 65535, 255, 0).astype(np.uint8)

    # Find contours in the segmentation mask
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour (assuming there is only one object)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # epsilon = 0.002 * cv2.arcLength(largest_contour, True)
    # approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    # segmentation_mask = np.zeros_like(image)
    # cv2.drawContours(segmentation_mask, [approx_contour], -1, (255, 255, 255), thickness=cv2.FILLED)


    # Calculate the center point of the bounding box
    center_x = x + w / 2
    center_y = y + h / 2

    # Determine which side of the bounding box is longer
    if w > h:
        side_length = w
    else:
        side_length = h

    # Set the length of the longer side to be equal to the length of the shorter side
    w = side_length
    h = side_length

    # Adjust the x and y values to ensure the center point remains the same
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)

    # Define the new quadratic bounding box
    new_box = (x, y, w, h)
    
    
    return new_box, segmentation_mask

def get_rendering(obj_model,rot_pose,tra_pose, ren):
    ren.clear()
    M=np.eye(4)
    M[:3,:3]=rot_pose
    M[:3,3]=tra_pose
    ren.draw_model(obj_model, M)
    img_r, depth_rend = ren.finish()
    img_r = img_r[:,:,::-1] *255
    vu_valid = np.where(depth_rend>0)
    bbox_gt = np.array([np.min(vu_valid[0]),np.min(vu_valid[1]),np.max(vu_valid[0]),np.max(vu_valid[1])])
    return img_r,depth_rend,bbox_gt


def get_sympose(rot_pose,sym):
    rotation_lock=False
    if(np.sum(sym)>0): #continous symmetric
        axis_order='s'
        multiply=[]
        for axis_id,axis in enumerate(['x','y','z']):
            if(sym[axis_id]==1):
                axis_order+=axis
                multiply.append(0)
        for axis_id,axis in enumerate(['x','y','z']):
            if(sym[axis_id]==0):
                axis_order+=axis
                multiply.append(1)

        axis_1,axis_2,axis_3 =tf3d.euler.mat2euler(rot_pose,axis_order)
        axis_1 = axis_1*multiply[0]
        axis_2 = axis_2*multiply[1]
        axis_3 = axis_3*multiply[2]            
        rot_pose =tf3d.euler.euler2mat(axis_1,axis_2,axis_3,axis_order) #
        sym_axis_tr = np.matmul(rot_pose,np.array([sym[:3]]).T).T[0]
        z_axis = np.array([0,0,1])
        #if symmetric axis is pallell to the camera z-axis, lock the rotaion augmentation
        inner = np.abs(np.sum(sym_axis_tr*z_axis))
        if(inner>0.8):
            rotation_lock=True #lock the in-plane rotation  
    
    return rot_pose,rotation_lock
    

    
if __name__=="__main__":
    root_dir = "./templates/tracebot"

    goal_dir = "./templates/tracebot_desc"

    gt_file = "./gts/tracebot_template_gt.json"

    with open(gt_file, "r") as f:
        tless_gt = json.load(f)

    # for folder in tqdm(os.listdir(root_dir)):
    #     print(folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # device='cpu'

    extractor = PoseViTExtractor(model_type='dino_vits8', stride=4, device=device)

    cam_K = np.array([909.9260864257812, 0.0, 643.5625,
                                0.0, 907.9168701171875, 349.0171813964844,
                                0.0, 0.0, 1.0]).reshape((3,3))

    ren = Renderer((1280, 720),cam_K)
    
    # already_done = ['02','30','22','17','04','13','21','06','19','24','08']
    numbers_to_exclude = []

    with torch.no_grad():

        for obj_id, labels in tqdm(tless_gt.items()):
            
            # print(obj_id)
            
            # if obj_id !='08':
            #     continue

            obj_model = Model3D()
            model_path = os.path.join(goal_dir,"models_xyz", f"obj_{int(obj_id):06d}.ply")
            # obj_model.load(model_path,scale=0.001)
            obj_model.load(model_path,scale=1.0)

            output_folder_path = os.path.join(goal_dir, obj_id)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            for i, label in enumerate(labels):
                
                # x, y, w, h = label['bbox_obj']
                
                # if i in [63,64,143,144,183,184,218,259,260,261,262,263,264,268,269,274]:
                # #if i == 63 or i==64 or i==143 or i==144 or i==183 or i ==184 or i==218 or i==259 or i==260 or i==261 or i==262 or i==263:
                #     continue
                # print(label)
                img_name = label['img_name'].split("/")[-1]
                img_id = img_name.split(".")[0]
                # print(img_name)

                img = cv2.imread(label['img_name'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                x, y, w, h = label["bbox_obj"]
                model_info = label["model_info"]

                # Calculate the size of the square crop
                size = max(w, h)

                # Calculate the center of the bounding box
                center_x = x + w / 2
                center_y = y + h / 2


                # Calculate the coordinates of the top-left corner of the square crop
                crop_x = int(center_x - size / 2)
                crop_y = int(center_y - size / 2)
                
                img_crop, crop_x, crop_y = img_utils.make_quadratic_crop(img, label['bbox_obj'])
                
                # if w >= 720 or h >= 540 or crop_y <=0 or crop_x <= 0 or crop_x+size>=img.shape[1] or crop_y+size>=img.shape[0]:
                #     numbers_to_exclude.append(i)
                #     continue
                
                # print(img.shape)
                # img_crop = img[crop_y:crop_y+size, crop_x:crop_x+size, :]
                
                # print(h, w, center_x, center_y, size)
                # print(crop_x, crop_y, size)
                # print(i)

                # print(img_crop.shape)
                img_prep, img_crop,_ = extractor.preprocess(Image.fromarray(img_crop), load_size=224)

                # desc = extractor.extract_descriptors(img_prep.to(device), layer=11, facet='key', bin=False, include_cls=True)

                # print(img_prep.shape)
                desc = extractor.extract_descriptors(img_prep.to(device), layer=11, facet='key', bin=False, include_cls=True)

                # saliency_map = extractor.extract_saliency_maps(img_prep.to(device))[0]

                # fg_mask1 = saliency_map > 0.05

                desc = desc.squeeze(0).squeeze(0).detach().cpu().numpy()

                R = np.array(label['cam_R_m2c']).reshape(3,3)
                t = np.array(label['cam_t_m2c'])
                cam_K = np.array(label['cam_K']).reshape(3,3)


                model_id = obj_id
                keys = model_info.keys()
                sym_continous = [0,0,0,0,0,0]
                if('symmetries_discrete' in keys):
                    pass
                    # print(model_id,"is symmetric_discrete")
                    # print("During the training, discrete transform will be properly handled by transformer loss")
                if('symmetries_continuous' in keys):
                    # print(model_id,"is symmetric_continous")
                    # print("During the rendering, rotations w.r.t to the symmetric axis will be ignored")
                    sym_continous[:3] = model_info['symmetries_continuous'][0]['axis']
                    sym_continous[3:]= model_info['symmetries_continuous'][0]['offset']

                rot_pose,rotation_lock = get_sympose(R,sym_continous)
                # img_uv, depth_rend, bbox_gt = get_rendering(obj_model,rot_pose,t/1000,ren)
                img_uv, depth_rend, bbox_gt = get_rendering(obj_model,rot_pose,t,ren)

                img_uv = img_uv.astype(np.uint8)

                img_uv, _, _ = img_utils.make_quadratic_crop(img_uv, [crop_y, crop_x, size, size])
                # img_uv = img_uv[crop_x:crop_x+size, crop_y:crop_y+size, :]

                # "img_crop": os.path.join(crop_folder_path, f"{int(templ_numb):06d}.png"),
                # "img_desc": os.path.join(desc_folder_path, f"{int(templ_numb):06d}.npy"),
                # "uv_crop": os.path.join(uv_map_folder_path, f"{int(templ_numb):06d}.npy") 

                tless_gt[obj_id][i]['img_crop'] = os.path.join(goal_dir, obj_id, img_name) 
                tless_gt[obj_id][i]['img_desc'] = os.path.join(goal_dir, obj_id, f"{img_id}.npy")
                tless_gt[obj_id][i]['uv_crop'] = os.path.join(goal_dir, obj_id, f"{img_id}_uv.npy")


                np.save(tless_gt[obj_id][i]['uv_crop'], img_uv)
                np.save(tless_gt[obj_id][i]['img_desc'], desc)
                img_crop.save(tless_gt[obj_id][i]['img_crop'])
    
    # print(numbers_to_exclude)
    with open("./gts/tracebot_templates_gt_uv.json","w") as f:
        json.dump(tless_gt, f)

        
    
    
    


