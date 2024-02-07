import os
import argparse
import json
import torch
from tqdm import tqdm
import numpy as np
from pose_extractor import PoseViTExtractor
from extractor import ViTExtractor
import copy
from pose_utils.data_utils import ImageContainer_masks
import pose_utils.img_utils as img_utils
from PIL import Image
import cv2
import pose_utils.utils as utils
import pose_utils.vis_utils as vis_utils
import time
import matplotlib.pyplot as plt
import pose_utils.eval_utils as eval_utils
import csv





if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    parser.add_argument('--config_file', default="./dino_pose_configs/cfg_lmo_inference_bop.json")

    args = parser.parse_args()
    
    
    debug = False
    img_name_to_debug = "000217"
    obj_name_to_debug = "can"

    with open(os.path.join(args.config_file), 'r') as f:
        config = json.load(f)
    
    # Loading ground truth files:

    with open(os.path.join(config['templates_gt_path']), 'r') as f:
        templates_gt = json.load(f)
    
    with open(os.path.join(config['gt_path']), 'r') as f:
        data_gt = json.load(f)
    
    with open(os.path.join(config['norm_factor_path']), 'r') as f:
        norm_factors = json.load(f)


    # Set up a results csv file:
    csv_file = f"./results/{config['results_file']}.csv"

    # Column names for the CSV file
    headers = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']

    # Create a new CSV file and write the headers
    with open(csv_file, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)
    
    # Create a debug_imgs folder:
    if config['debug_imgs']:
        if not os.path.exists(f"./dbg_imgs/{config['results_file']}"):
            os.makedirs(f"./dbg_imgs/{config['results_file']}")
        else:
            print(f"./dbg_imgs/{config['results_file']} already exists!")


    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    extractor = PoseViTExtractor(model_type='dino_vits8', stride=4, device=device)


    # Loading templates into gpu
    templates_desc = {}
    templates_crops = {}
    tmpdic_per_obj = {}
    templates_gt_subset = {}
    for obj_id, template_labels in tqdm(templates_gt.items()):
        templates_desc[obj_id] = torch.cat([torch.from_numpy(np.load(template_label['img_desc'])).unsqueeze(0)
                                              for i, template_label in enumerate(template_labels) 
                                              if i%config['subset_templates']==0], dim=0)
        
        templates_gt_subset[obj_id] = [template_label for i, template_label in 
                                    enumerate(template_labels) if i%config['subset_templates']==0]
        
    
    print("Preparing templates and loading of extractor is done!")
    
    

    for all_id, img_labels in tqdm(data_gt.items()):
        scene_id = all_id.split("_")[0]
        img_id = all_id.split("_")[-1]
        
        # get data and crops for a single image
        
        img_path = os.path.join(config['dataset_path'], img_labels[0]['img_name'].split("./")[-1])
        img_name = img_path.split("/")[-1].split(".png")[0]
        
        
        if debug:
            if img_name != img_name_to_debug:
                continue
        
        img = Image.open(img_path)
        cam_K = np.array(img_labels[0]['cam_K']).reshape((3,3))

        img_data = ImageContainer_masks(img = img,
                                  img_name = img_name,
                                  scene_id = img_labels[0]['scene_id'],
                                  cam_K = cam_K, 
                                  crops = [],
                                  descs = [],
                                  x_offsets = [],
                                  y_offsets = [],
                                  obj_names = [],
                                  obj_ids = [],
                                  model_infos = [],
                                  t_gts = [],
                                  R_gts = [],
                                  masks = [])

        for obj_index, img_label in enumerate(img_labels):
            bbox_gt = img_label[config['bbox_type']]

            if bbox_gt[2] == 0 or bbox_gt[3] == 0:
                continue

            if bbox_gt != [-1,-1,-1,-1]:
                img_data.t_gts.append(np.array(img_label['cam_t_m2c']) * config['scale_factor'])
                img_data.R_gts.append(np.array(img_label['cam_R_m2c']).reshape((3,3)))
                img_data.obj_names.append(img_label['obj_name'])
                img_data.obj_ids.append(str(img_label['obj_id']))
                img_data.model_infos.append(img_label['model_info'])


                try:
                    mask = img_utils.rle_to_mask(img_label['mask_sam'])
                    
                    mask = mask.astype(np.uint8)
                    
                    mask_3_channel = np.stack([mask] * 3, axis=-1)
                    
                    bbox = img_utils.get_bounding_box_from_mask(mask)
                    
                    img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
                    
                    mask_crop,_,_ = img_utils.make_quadratic_crop(mask, bbox)

                    img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)

                    img_data.crops.append(Image.fromarray(img_crop))

                    img_prep, img_crop,_ = extractor.preprocess(Image.fromarray(img_crop), load_size=224)

                    with torch.no_grad():
                        desc = extractor.extract_descriptors(img_prep.to(device), layer=11, facet='key', bin=False, include_cls=True)
                        img_data.descs.append(desc.squeeze(0).squeeze(0).detach().cpu())
                        '''
                        # extract_multi_descriptors
                        start_desc_ex = time.time()
                        img_prep = img_prep.to(device)
                        descs = extractor.extract_multi_descriptors(img_prep.to(device))
                        # desc1 = extractor.extract_descriptors(img_prep, layer=11, facet='key', bin=False, include_cls=True)
                        # desc = extractor.extract_descriptors(img_prep, layer=9, facet='key', bin=True, include_cls=False)
                        # img_data.descs.append(desc.squeeze(0).squeeze(0).detach().cpu())
                        end_desc_ex = time.time()
                        elapsed_desc_ex = end_desc_ex - start_desc_ex
                        
                        img_data.descs.append(descs[1].squeeze(0).squeeze(0).detach().cpu())
                        
                        
                        # desc = extractor.extract_descriptors(img_prep.to(device), layer=11, facet='key', bin=False, include_cls=True)
                        # img_data.descs.append(desc.squeeze(0).squeeze(0).detach().cpu())
                        '''
                    
                    img_data.y_offsets.append(y_offset)
                    img_data.x_offsets.append(x_offset)
                    img_data.masks.append(mask_3_channel)
                        
                except:
                    print("No segmentation mask found!")
                    img_data.crops.append(None)
                    img_data.descs.append(None)
                    img_data.y_offsets.append(None)
                    img_data.x_offsets.append(None)
                    img_data.masks.append(None)

        
        for i in range(len(img_data.crops)):
            
            object_id = img_data.obj_ids[i]
            if img_data.crops[i] is not None:
                start_time_match = time.time()
                matched_templates = utils.find_template_cpu(img_data.descs[i], 
                                                            templates_desc[img_data.obj_names[i]], 
                                                            num_results=config['num_matched_templates'])
                end_time_match = time.time()
                elapsed_time_match = end_time_match - start_time_match
                
                start_time = time.time()
                min_err = np.inf
                pose_est = False
                
                start_all_obj = time.time()
                
                for matched_template in matched_templates:

                    template = Image.open(templates_gt_subset[img_data.obj_names[i]][matched_template[1]]['img_crop'])


                    # try:
                    if True:
                        with torch.no_grad():
                            # image2_batch, image2_pil, scale_factor = extractor.preprocess(template, load_size=img_data.crops[i].size[0])
                            # descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer = 9, facet = 'key', bin = True)
                            start_time_fc = time.time()
                            # points1, points2, crop_pil, template_pil = extractor.find_correspondences_fastknn_old(img_data.crops[i], 
                            #                                                                             template, 
                            #                                                                             prep_img2=image2_batch,
                            #                                                                             desc_img2=descriptors2,
                            #                                                                             num_pairs=config['num_correspondences'],
                            #                                                                             load_size=img_data.crops[i].size[0])
                            
                            points1, points2, crop_pil, template_pil = extractor.find_correspondences_fastknn_old(img_data.crops[i], 
                                                                                                        template, 
                                                                                                        num_pairs=20,
                                                                                                        load_size=img_data.crops[i].size[0])
                            
                            end_time_fc = time.time()
                            elapsed_time_fc = end_time_fc - start_time_fc
                         

                        start_time_pose = time.time()
                        img_uv = np.load(f"{templates_gt_subset[img_data.obj_names[i]][matched_template[1]]['img_crop'].split('.png')[0]}_uv.npy")
                        
                        img_uv = img_uv.astype(np.uint8)

                        img_uv = cv2.resize(img_uv, img_data.crops[i].size)

                        R_est, t_est = utils.get_pose_from_correspondences(points1,
                                                                        points2,
                                                                        img_data.y_offsets[i],
                                                                        img_data.x_offsets[i],
                                                                        img_uv,
                                                                        img_data.cam_K,
                                                                        norm_factors[str(img_data.obj_ids[i])],
                                                                        config['scale_factor'])
                        end_time_pose = time.time()
                        elapsed_time_pose = end_time_pose - start_time_pose
                        
                    # except:
                    #     print("Something went wrong during find_correspondences!")
                    #     print(i)
                    #     R_est = None
                    

                    if R_est is None:
                        print(f"Not enough feasible correspondences where found for {img_data.obj_ids[i]}")
                        R_est = np.array(templates_gt_subset[img_data.obj_names[i]][matched_template[1]]['cam_R_m2c']).reshape((3,3))
                        t_est = np.array([0.,0.,0.])

                    end_time = time.time()

                    if img_data.obj_ids[i] == "eggbox" or img_data.obj_ids[i] == "glue":
                        err, acc = eval_utils.calculate_score(R_est, img_data.R_gts[i], int(img_data.obj_ids[i]), 0)
                    else:
                        err, acc = eval_utils.calculate_score(R_est, img_data.R_gts[i], int(img_data.obj_ids[i]), 0)

                    if err < min_err:
                        min_err = err
                        R_best = R_est
                        t_best = t_est
                        elapsed_time = end_time-start_time
                        pose_est = True
                
                if not pose_est:
                    R_best = np.array([[1.0,0.,0.],
                                    [0.,1.0,0.],
                                    [0.,0.,1.0]])
                    
                    t_best = np.array([0.,0.,0.])
                    print("No Pose could be determined")
                    score = 0.
                else:
                    score = 1.0
            
            else:
                R_best = np.array([[1.0,0.,0.],
                [0.,1.0,0.],
                [0.,0.,1.0]])
                    
                t_best = np.array([0.,0.,0.])
                print("No Pose could be determined")
                score = 0.

                
                
            # Prepare for writing:
            R_best_str = " ".join(map(str, R_best.flatten()))
            t_best_str = " ".join(map(str, t_best * 1000))
            elapsed_time = -1
            
            end_time_all = time.time()
            
            elapsed_all = end_time_all - start_all_obj
            
            # Write the detections to the CSV file
            print(f"time_per_obj: {elapsed_all}")
            
            # print(f"match_time: {elapsed_time_match}, corr_time: {elapsed_time_fc}, pose_time: {elapsed_time_pose}")
            
            # ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
            with open(csv_file, mode='a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([img_data.scene_id, img_data.img_name, object_id, score, R_best_str, t_best_str, elapsed_time])


            if config['debug_imgs']:
                if int(img_id) % config['debug_imgs'] == 0:
                    dbg_img = vis_utils.create_debug_image(R_best, t_best, img_data.R_gts[i], img_data.t_gts[i], np.asarray(img_data.img),
                                        img_data.cam_K, img_data.model_infos[i],
                                        config['scale_factor'])
                    
                    dbg_img = cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB)
                    if img_data.masks[i] is not None:
                        dbg_img_mask = cv2.hconcat([dbg_img, img_data.masks[i]]) 
                    else:
                        dbg_img_mask = dbg_img   
                        
                    cv2.imwrite(f"./dbg_imgs/{config['results_file']}/{img_data.img_name}_{img_data.obj_ids[i]}.png", dbg_img_mask)
            

                