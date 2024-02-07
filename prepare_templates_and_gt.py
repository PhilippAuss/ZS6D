from rendering.renderer_xyz import Renderer
from rendering.model import Model3D
from rendering import utils as renderutil
import os
import json
import cv2
import numpy as np
import transforms3d as tf3d
from pose_utils import vis_utils
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
    parser = argparse.ArgumentParser(description='Prepare templates, calculate descriptors, uv_maps and ground truth file')
    parser.add_argument('--config_file', default="./dino_pose_configs/cfg_tracebot_and_gt")
    
    debug = False
    obj_to_debug = "duck"
    templ_numb_to_debug = "2"
    
    args = parser.parse_args()

    with open(os.path.join(args.config_file), 'r') as f:
        config = json.load(f)
    
    obj_name_id = config['obj_name_id_dict']
    root_dir = config['templates_root_dir']
    dataset = config['dataset_id']
    
    obj_id_name = {v: k for k, v in obj_name_id.items()}
    
    with open(os.path.join(root_dir,"models","models_info.json"),"r") as f:
        model_infos = json.load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #Preparing output file:
    template_labels_dict = {}

    # extractor = PoseViTExtractor(model_type='dino_vits8', stride=4, device=device)
    
    extractor = PoseViTExtractor(model_type='dino_vits8', stride=2, device=device)
    
    for folder in tqdm(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        
        
        # Make sure it's a folder and it starts with dataset
        if os.path.isdir(folder_path) and folder.startswith(dataset):
            train_pbr_path = os.path.join(folder_path, 'train_pbr')

            obj_id = folder_path.split(dataset)[-1]
            
            if debug:
                if obj_id not in ['2','3','4','7','13','14','15']: 
                    if obj_id_name[obj_id] != obj_to_debug:
                        continue
            
            obj_model = Model3D()
            model_path = os.path.join(root_dir,"models_xyz", f"obj_{int(obj_id):06d}.ply")

            if not os.path.exists(model_path):
                continue

            obj_model.load(model_path,scale=0.001)
            
            # Define the template_list for one object type:
            template_list = []
            counter = 0
            for subfolder in os.listdir(train_pbr_path):
                    subfolder_path = os.path.join(train_pbr_path, subfolder)

                    if os.path.isdir(subfolder_path):
                        scene_gt_path = os.path.join(subfolder_path, 'scene_gt.json')
                        scene_camera_path = os.path.join(subfolder_path, 'scene_camera.json')
                        
                        
                        
                        # Check if folders exists otherwise create them:
                        crop_folder_path = os.path.join(subfolder_path, 'crop')
                        if not os.path.exists(crop_folder_path):
                            os.makedirs(crop_folder_path)
                        
                        mask_folder_path = os.path.join(subfolder_path,'mask')
                        if not os.path.exists(mask_folder_path):
                            os.makedirs(mask_folder_path)
                        
                        mask_crop_folder_path = os.path.join(subfolder_path,'crop_mask')
                        if not os.path.exists(mask_crop_folder_path):
                            os.makedirs(mask_crop_folder_path)
                            
                        uv_map_folder_path = os.path.join(subfolder_path,'uv_map')
                        if not os.path.exists(uv_map_folder_path):
                            os.makedirs(uv_map_folder_path)
                            
                        desc_folder_path = os.path.join(subfolder_path,'desc')
                        if not os.path.exists(desc_folder_path):
                            os.makedirs(desc_folder_path)
                            
                        
                        with open(scene_gt_path, 'r') as gt_file:
                            gt_data = json.load(gt_file)
                        
                        with open(scene_camera_path, 'r') as camera_file:
                            camera_data = json.load(camera_file)
                        
                        cam_K = np.array(camera_data['0']['cam_K']).reshape(3,3) 
                           
                        # Hardcoded have to change:    
                        ren = Renderer((1280,720),cam_K)
                        
                        for templ_numb, cam_numb in zip(gt_data, camera_data):
                            templ_data = gt_data[templ_numb][0]
                            cam_data = camera_data[cam_numb]
                            
                            templ_path = os.path.join(subfolder_path, "rgb", f"{int(templ_numb):06d}.jpg")
                            depth_path = os.path.join(subfolder_path, "depth", f"{int(templ_numb):06d}.png")
                            templ_path_crop = os.path.join(subfolder_path, "rgb", 
                                                        f"{int(templ_numb):06d}_template.png")
                            
                            bbox, mask = get_bbox_and_segmentation(depth_path)
                            x,y,w,h = bbox
                            cropped_mask = mask[y:y+h, x:x+w]
                            
                            templ = cv2.imread(templ_path)
                            templ = cv2.cvtColor(templ, cv2.COLOR_BGR2RGB)
                            cropped_templ = templ[y:y+h, x:x+w]

                            
                            "-------------------------------------------------------------------------------------------"
                            with torch.no_grad():
                            
                                img_prep, img_crop,_ = extractor.preprocess(Image.fromarray(cropped_templ), load_size=224)

                                # desc = extractor.extract_descriptors(img_prep.to(device), layer=11, facet='key', bin=False, include_cls=True)
                                
                                desc = extractor.extract_descriptors(img_prep.to(device), layer=9, facet='key', bin=False, include_cls=False)
                                
                                # saliency_map = extractor.extract_saliency_maps(img_prep.to(device))[0]
                                
                                # fg_mask1 = saliency_map > 0.05

                                desc = desc.squeeze(0).squeeze(0).detach().cpu().numpy()
                            
                            "-------------------------------------------------------------------------------------------"
                            
                            if debug and obj_id_name[str(obj_id)] == obj_to_debug and templ_numb != templ_numb_to_debug:
                                hello=1
                            
                            R = np.array(templ_data['cam_R_m2c']).reshape(3,3)
                            t = np.array(templ_data['cam_t_m2c'])
                            cam_K = np.array(cam_data['cam_K']).reshape(3,3)
                            model_info = model_infos[obj_id]
                            
                            
                            
                            
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
                            img_uv, depth_rend, bbox_gt = get_rendering(obj_model,rot_pose,t/1000,ren)
                            img_uv = img_uv[y:y+h, x:x+w, :]
                            
                            
                            template_dic = {"img_id": str(counter),
                                            "img_name": templ_path,
                                            "mask_name": os.path.join(mask_folder_path,f"{int(templ_numb):06d}.png"),
                                            "crop_mask_name": os.path.join(mask_crop_folder_path,f"{int(templ_numb):06d}.png"),
                                            "bbox_obj": bbox,
                                            "cam_R_m2c": R.tolist(),
                                            "cam_t_m2c": t.tolist(),
                                            "model_xyz_path": model_path,
                                            "model_path": os.path.join(root_dir,"models", f"obj_{int(obj_id):06d}.ply"),
                                            "model_info": model_info,
                                            "cam_K": cam_K.tolist(),
                                            "img_crop": os.path.join(crop_folder_path, f"{int(templ_numb):06d}.png"),
                                            "img_desc": os.path.join(desc_folder_path, f"{int(templ_numb):06d}.npy"),
                                            "uv_crop": os.path.join(uv_map_folder_path, f"{int(templ_numb):06d}.npy")                                            
                            }
                            
                            np.save(template_dic['uv_crop'], img_uv)
                            np.save(template_dic['img_desc'], desc)
                            cv2.imwrite(template_dic['mask_name'], mask)
                            cv2.imwrite(template_dic['crop_mask_name'], cropped_mask)
                            img_crop.save(template_dic['img_crop'])
                            # cv2.imwrite(template_dic['img_crop'], cropped_templ)
                            
                            template_list.append(template_dic)
                            
                            counter += 1
                            
                            
                            
                            
                            
                            
            template_labels_dict[obj_id_name[str(obj_id)]] = template_list                
            
            with open(os.path.join("./gts", config['gt_name']), 'w') as file:
                json.dump(template_labels_dict, file)
                      
                            
                            
                            
                            
                            
                            
        #                     plt.imshow(img_uv.astype(np.uint8))
        #                     plt.show()
                                        
        #                     out_img = vis_utils.draw_3D_bbox_on_image(templ, R, t, cam_K, 
        #                                                                 model_info, 
        #                                                                 image_shape=templ.shape, 
        #                                                                 factor=1, #0.001, 
        #                                                                 colEst=(0, 205, 205))
                            
        #                     #out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        #                     plt.imshow(out_img)
        #                     plt.show()
                            
        #                     break

        #             break
        # break
        
    
    
    


