import argparse
import os
import json
import numpy as np
import torch
from pose_extractor import PoseViTExtractor
from tools.ply_file_to_3d_coord_model import convert_unique
from rendering.renderer_xyz import Renderer
from rendering.model import Model3D
from tqdm import tqdm
import cv2
from PIL import Image
from pose_utils import vis_utils
from pose_utils import img_utils
from rendering.utils import get_rendering, get_sympose


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    parser.add_argument('--config_file', default="./zs6d_configs/template_gt_preparation_configs/cfg_template_gt_generation_ycbv.json")

    args = parser.parse_args()
    
    with open(os.path.join(args.config_file),'r') as f:
        config = json.load(f)
        
    with open(os.path.join(config['path_models_info_json']), 'r') as f:
        models_info = json.load(f)
    
    obj_poses = np.load(config['path_template_poses'])
    
    # Creating the output folder for the cropped templates and descriptors
    if not os.path.exists(config['path_output_templates_and_descs_folder']):
        os.makedirs(config['path_output_templates_and_descs_folder'])
       
    # Creating the models_xyz folder
    if not os.path.exists(config['path_output_models_xyz']):
        os.makedirs(config['path_output_models_xyz'])
    
    # Preparing the object models in xyz format:
    print("Loading and preparing the object meshes:")
    norm_factors = {}
    for obj_model_name in tqdm(os.listdir(config['path_object_models_folder'])):
        if obj_model_name.endswith(".ply"):
            obj_id = int(obj_model_name.split("_")[-1].split(".ply")[0])
            input_model_path = os.path.join(config['path_object_models_folder'], obj_model_name)
            output_model_path = os.path.join(config['path_output_models_xyz'], obj_model_name)
            if not os.path.exists(output_model_path):
                x_abs,y_abs,z_abs,x_ct,y_ct,z_ct = convert_unique(input_model_path, output_model_path)
                
                norm_factors[obj_id] = {'x_scale':float(x_abs),
                                       'y_scale':float(y_abs),
                                       'z_scale':float(z_abs),
                                       'x_ct':float(x_ct),
                                       'y_ct':float(y_ct),
                                       'z_ct':float(z_ct)}
    
    with open(os.path.join(config['path_output_models_xyz'],"norm_factor.json"),"w") as f:
        json.dump(norm_factors,f)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    extractor = PoseViTExtractor(model_type='dino_vits8', stride=4, device=device)
    
    cam_K = np.array(config['cam_K']).reshape((3,3))
    
    ren = Renderer((config['template_resolution'][0], config['template_resolution'][1]), cam_K)
    
    template_labels_gt = dict()
    
    with torch.no_grad():
        
        for template_name in tqdm(os.listdir(config['path_templates_folder'])):
            
            path_template_folder = os.path.join(config['path_templates_folder'], template_name)
            
            if os.path.isdir(path_template_folder) and template_name != "models" and template_name != "models_proc":
                
                path_to_template_desc = os.path.join(config['path_output_templates_and_descs_folder'],
                                                     template_name)
                
                if not os.path.exists(path_to_template_desc):
                    os.makedirs(path_to_template_desc)
                
                obj_id = template_name.split("_")[-1]
                
                model_info = models_info[str(obj_id)]
                
                obj_model = Model3D()
                model_path = os.path.join(config['path_output_models_xyz'], f"obj_{int(obj_id):06d}.ply")
                
                # Some objects are scaled inconsistently within the dataset, these exceptions are handled here:
                obj_scale = config['obj_models_scale']
                obj_model.load(model_path, scale=obj_scale)
                
                files = os.listdir(path_template_folder)
                filtered_files = list(filter(lambda x: not x.startswith('mask_'), files))
                filtered_files.sort(key=lambda x: os.path.getmtime(os.path.join(path_template_folder,x)))
                
                tmp_list = []
                
                for i, file in enumerate(filtered_files):
                    
                    # Preparing mask and bounding box [x,y,w,h]
                    mask_path = os.path.join(path_template_folder, f"mask_{file}")
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    x, y, w, h = cv2.boundingRect(contours[0])
                    crop_size = max(w,h)
                    
                    # Preparing cropped image and desc
                    img = cv2.imread(os.path.join(path_template_folder, file))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_crop, crop_x, crop_y = img_utils.make_quadratic_crop(img, [x, y, w, h])
                    img_prep, img_crop, _ = extractor.preprocess(Image.fromarray(img_crop), load_size=224)
                    desc = extractor.extract_descriptors(img_prep.to(device), layer=11, facet='key', bin=False, include_cls=True)
                    desc = desc.squeeze(0).squeeze(0).detach().cpu().numpy()
                    
                    # R = obj_poses[i].T[:3,:3].T
                    R = obj_poses[i][:3,:3]
                    t = obj_poses[i].T[-1,:3] / obj_scale
                    sym_continues = [0,0,0,0,0,0]
                    keys = model_info.keys()
                    
                    if('symmetries_continuous' in keys):
                        sym_continues[:3] = model_info['symmetries_continuous'][0]['axis']
                        sym_continues[3:] = model_info['symmetries_continuous'][0]['offset']
                    
                    rot_pose, rotation_lock = get_sympose(R, sym_continues)

                    img_uv, depth_rend, bbox_template = get_rendering(obj_model, rot_pose, t, ren)

                    img_uv = img_uv.astype(np.uint8)
                    
                    img_uv,_,_ = img_utils.make_quadratic_crop(img_uv, [crop_y, crop_x, crop_size, crop_size])
                    
                    
                    # Storing template information:
                    tmp_dict = {"img_id": str(i),
                                "img_name":os.path.join(os.path.join(path_template_folder,file)),
                                "mask_name":os.path.join(os.path.join(path_template_folder,f"mask_{file}")),
                                "obj_id": str(obj_id),
                                "bbox_obj": [x,y,w,h],
                                "cam_R_m2c": R.tolist(),
                                "cam_t_m2c": t.tolist(),
                                "model_path": os.path.join(config['path_object_models_folder'], f"obj_{int(obj_id):06d}.ply"),
                                "model_info": models_info[str(obj_id)],
                                "cam_K": cam_K.tolist(),
                                "img_crop": os.path.join(path_to_template_desc, file),
                                "img_desc": os.path.join(path_to_template_desc, f"{file.split('.')[0]}.npy"),
                                "uv_crop": os.path.join(path_to_template_desc, f"{file.split('.')[0]}_uv.npy"),
                        
                    }
                    
                    tmp_list.append(tmp_dict)
                    
                    # Saving all template crops and descriptors:
                    np.save(tmp_dict['uv_crop'], img_uv)
                    np.save(tmp_dict['img_desc'], desc)
                    img_crop.save(tmp_dict['img_crop'])

                
                template_labels_gt[str(obj_id)] = tmp_list
            
    with open(config['output_template_gt_file'], 'w') as f:
        json.dump(template_labels_gt, f)   
                    
                    
                    
                    
                    
                    

                
    
    
    
    
    