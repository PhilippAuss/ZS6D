import os
import json
import torch
from tqdm import tqdm
import numpy as np
from pose_extractor import PoseViTExtractor
import pose_utils.img_utils as img_utils
from PIL import Image
import cv2
import pose_utils.utils as utils
import time




class ZS6D:
    
    def __init__(self, templates_gt_path, norm_factors_path, model_type='dino_vits8', stride=4, subset_templates=1, max_crop_size=80):
        
        self.model_type = model_type
        self.stride = stride
        self.subset_templates = subset_templates
        self.max_crop_size = max_crop_size
        
        with open(os.path.join(templates_gt_path),'r') as f:
            self.templates_gt = json.load(f)
        
        with open(os.path.join(norm_factors_path), 'r') as f:
            self.norm_factors = json.load(f)
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.extractor = PoseViTExtractor(model_type = self.model_type, stride = self.stride, device = self.device)
        
        self.templates_desc = {}
        templates_gt_subset = {}
        for obj_id, template_labels in tqdm(self.templates_gt.items()):
            self.templates_desc[obj_id] = torch.cat([torch.from_numpy(np.load(template_label['img_desc'])).unsqueeze(0)
                                                for i, template_label in enumerate(template_labels) 
                                                if i%subset_templates==0], dim=0)
            
            templates_gt_subset[obj_id] = [template_label for i, template_label in 
                                        enumerate(template_labels) if i%subset_templates==0]
        
        self.templates_gt = templates_gt_subset
        
        print("Preparing templates and loading of extractor is done!")
        
        
    def get_pose(self, img, obj_name, obj_id, mask, cam_K, bbox=None):
        
        if bbox is None:
            bbox = img_utils.get_bounding_box_from_mask(mask)
        
        img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
        
        mask_crop,_,_ = img_utils.make_quadratic_crop(mask, bbox)
        
        img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
        
        img_crop = Image.fromarray(img_crop) 
        
        img_prep, _, _ = self.extractor.preprocess(img_crop, load_size=224)
        
        with torch.no_grad():
            desc = self.extractor.extract_descriptors(img_prep.to(self.device), layer=11, facet='key', bin=False, include_cls=True)
            desc = desc.squeeze(0).squeeze(0).detach().cpu()
        
        matched_templates = utils.find_template_cpu(desc,
                                                    self.templates_desc[obj_name],
                                                    num_results=1)
        
        
        template = Image.open(self.templates_gt[obj_name][matched_templates[0][1]]['img_crop'])
        
        try:
            with torch.no_grad():
                
                if img_crop.size[0] < self.max_crop_size:
                    crop_size = img_crop.size[0]
                else:
                    crop_size = self.max_crop_size
                    
                    
                points1, points2, crop_pil, template_pil = self.extractor.find_correspondences_fastkmeans(img_crop, 
                                                                                                           template, 
                                                                                                           num_pairs=20,
                                                                                                           load_size=crop_size)
                
                
                img_uv = np.load(f"{self.templates_gt[obj_name][matched_templates[0][1]]['img_crop'].split('.png')[0]}_uv.npy")
                
                img_uv = img_uv.astype(np.uint8)
                
                img_uv = cv2.resize(img_uv, img_crop.size)
                
                R_est, t_est = utils.get_pose_from_correspondences(points1,
                                                                  points2,
                                                                  y_offset,
                                                                  x_offset,
                                                                  img_uv,
                                                                  cam_K,
                                                                  self.norm_factors[str(obj_id)],
                                                                  scale_factor=1)
                
                return R_est, t_est
        except:
            print("Error while local correspondence matching")
            
            return None, None
        

# if __name__=="__main__":
#     img_id = '3'
    
#     with open(os.path.join("./dino_pose_configs/cfg_lmo_inference_bop.json"), "r") as f:
#         config = json.load(f)
    

#     pose_estimator = ZS6D(config['templates_gt_path'], config['norm_factor_path'])
    
#     with open(os.path.join(config['gt_path']), 'r') as f:
#         data_gt = json.load(f)
        
#     for i in range(5):
#         obj_number = i

#         obj_name = data_gt[img_id][obj_number]['obj_name']
#         obj_id = data_gt[img_id][obj_number]['obj_id']
#         cam_K = np.array(data_gt[img_id][obj_number]['cam_K']).reshape((3,3))
#         bbox = data_gt[img_id][obj_number]['bbox_visib']

#         img_path = os.path.join(config['dataset_path'], data_gt[img_id][obj_number]['img_name'].split("./")[-1])
#         img = Image.open(img_path)

#         mask = data_gt[img_id][obj_number]['mask_sam']
#         mask = img_utils.rle_to_mask(mask)
#         mask = mask.astype(np.uint8)

#         start_time = time.time()

#         R_est, t_est = pose_estimator.get_pose(img, obj_name, obj_id, mask, cam_K, bbox=None)

#         end_time = time.time()

#         print(f"Pose estimation time: {end_time-start_time}")
#         print(f"R_est: {R_est}")
#         print(f"t_est: {t_est}")
        
        