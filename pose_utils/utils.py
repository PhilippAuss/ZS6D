import torch
import cv2
import numpy as np
from torchvision import transforms
from typing import Union, List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from src.correspondences import chunk_cosine_sim
from scipy.spatial.distance import cosine

def find_template(desc_input, desc_templates, num_results):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    desc_input = desc_input.to(device)
    desc_templates = desc_templates.to(device)
    
    similarities = [(1 - torch.nn.functional.cosine_similarity(desc_input.flatten(), desc_template.flatten(), dim=0).detach().cpu(), i) 
                    for i, desc_template in enumerate(desc_templates) 
                    if desc_input.shape == desc_template.shape]
    
    sorted_sims = sorted(similarities, key=lambda x: x[0], reverse=True)
    
    result = [(sim[0], sim[1]) for sim in sorted_sims[:num_results]]
    

    desc_input = desc_input.detach().to("cpu")
    desc_templates = desc_templates.detach().to("cpu")

    # # Clear GPU memory
    torch.cuda.empty_cache()
    
    return result


def find_template_cpu(desc_input, desc_templates, num_results):
    # Flatten and normalize the desc_input
    desc_input_flat = desc_input.ravel()
    desc_input_norm = np.linalg.norm(desc_input_flat)

    # Precompute flattening and norms for all templates
    templates_flat = [template.ravel() for template in desc_templates]
    templates_norm = [np.linalg.norm(template_flat) for template_flat in templates_flat]

    # Compute cosine similarities in a vectorized manner
    similarities = [(np.dot(desc_input_flat, template_flat) / (desc_input_norm * template_norm), i)
                    for i, (template_flat, template_norm) in enumerate(zip(templates_flat, templates_norm))]

    # Sort the results
    sorted_sims = sorted(similarities, key=lambda x: x[0], reverse=True)

    # Return the top num_results
    return sorted_sims[:num_results]


def find_template_cpu_matrix(desc_input, desc_templates, num_results):
    # Flatten and normalize the desc_input
    desc_input_flat = desc_input.ravel()
    desc_input_norm = np.linalg.norm(desc_input_flat)

    # Convert list of templates to a 3D NumPy array and flatten along the last two dimensions
    templates_array = np.array(desc_templates).reshape(len(desc_templates), -1)
    templates_norms = np.linalg.norm(templates_array, axis=1)

    # Compute cosine similarities using matrix operations
    similarities = np.dot(templates_array, desc_input_flat) / (templates_norms * desc_input_norm)

    # Get the indices of the top num_results similarities
    top_indices = np.argsort(similarities)[-num_results:][::-1]

    # Return the top similarities and their indices
    return [(similarities[i], i) for i in top_indices]


def preprocess(img: Image.Image, 
               load_size: Union[int, Tuple[int, int]] = None,
               mean = (0.485, 0.456, 0.406),
               std = (0.229, 0.224, 0.225)) -> Tuple[torch.Tensor, Image.Image]:
    
    scale_factor = 1
    
    if load_size is not None:
        width, height = img.size # img has to be quadratic

        img = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(img)

        scale_factor = img.size[0]/width

    prep = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=mean, std=std)
                             ])
    
    prep_img = prep(img)[None, ...]

    return prep_img, img, scale_factor


def _transform_to_xyz(r, g, b, x_ct, y_ct, z_ct, x_scale, y_scale, z_scale):
    x = r / 255.
    x = x * 2 - 1
    x = x * x_scale + x_ct
    y = g / 255.
    y = y * 2 - 1
    y = y * y_scale + y_ct
    z = b / 255.
    z = z * 2 - 1
    z = z * z_scale + z_ct
    
    return x, y, z



def transform_2D_3D(points, img_uv, norm_factor):
    x_ct = norm_factor["x_ct"]
    y_ct = norm_factor["y_ct"]
    z_ct = norm_factor["z_ct"]
    x_scale = norm_factor["x_scale"]
    y_scale = norm_factor["y_scale"]
    z_scale = norm_factor["z_scale"]
    
    points_3D = []
    
    for point in points:
        r, g, b = img_uv[point[0],point[1]]
        x, y, z = _transform_to_xyz(r, g, b, x_ct, y_ct, z_ct, x_scale, y_scale, z_scale)
        points_3D.append([x,y,z])
    
    return points_3D

def weighted_solve_pnp_ransac(object_points, image_points, camera_matrix, dist_coeffs, weights, iterations=100, reprojection_error=8.0):
    best_inliers = []
    best_inlier_count = 0

    num_points = len(image_points)

    for _ in range(iterations):
        sample_indices = np.random.choice(num_points, 4, p=weights, replace=False)
        sampled_object_points = object_points[sample_indices]
        sampled_image_points = image_points[sample_indices]

        success, rvec, tvec = cv2.solvePnP(sampled_object_points, sampled_image_points, camera_matrix, dist_coeffs)

        if not success:
            continue

        projected_image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        error = np.linalg.norm(projected_image_points - image_points, axis=2).reshape(-1)
        weighted_error = error * weights
        inliers = np.where(weighted_error < reprojection_error)[0]

        inlier_count = len(inliers)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            best_rvec = rvec
            best_tvec = tvec

    refined_rvec, refined_tvec = cv2.solvePnP(object_points[best_inliers], image_points[best_inliers], camera_matrix, dist_coeffs)[1:3]

    return refined_rvec, refined_tvec, best_inliers


def get_pose_from_correspondences(points1, points2, y_offset, x_offset, img_uv, cam_K, norm_factor, scale_factor, resize_factor=1.0):
    
    # filter valid points
    valid_points1 = []
    valid_points2 = []
    for point1, point2 in zip(points1, points2):
        if np.any(img_uv[point2[0], point2[1]] != [0,0,0]):
            valid_points1.append(point1)
            valid_points2.append(point2)
    
    # Check if enough correspondences for PnPRansac
    if len(valid_points1) < 4:
        return None, None
    
    points2_3D = transform_2D_3D(valid_points2, img_uv, norm_factor)

    valid_points1 = np.array(valid_points1).astype(np.float64)/resize_factor

    valid_points1[:,0] += y_offset
    valid_points1[:,1] += x_offset

    valid_points1[:,[0,1]] = valid_points1[:,[1,0]]

    try:
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(points2_3D).astype(np.float64), valid_points1, cam_K,
                                                        distCoeffs=None, iterationsCount=100, reprojectionError=8.0)
    
    except:
        print("Solving PnP failed!")
        return None, None
    
    R_est, _ = cv2.Rodrigues(rvec)
    t_est = np.squeeze(tvec)*scale_factor

    return R_est, t_est


# def get_pose_from_correspondences_v2(list_points1, list_points2, y_offset, x_offset, list_img_uv, 
#                                      cam_K, norm_factor, scale_factor, resize_factor=1.0):
    
#     # Initialize empty lists for valid points
#     valid_points1 = []
#     points2_3D = []

#     # Iterate through multiple sets of points1, points2 and img_uv
#     for points1, points2, img_uv in zip(list_points1, list_points2, list_img_uv):

#         # filter valid points
#         valid_points2 = []
#         for point1, point2 in zip(points1, points2):
#             if np.any(img_uv[point2[0], point2[1]] != [0,0,0]):
#                 valid_points1.append(point1)
#                 valid_points2.append(point2)
        
#         points2_3D.extend(transform_2D_3D(valid_points2, img_uv, norm_factor))

#     # Check if enough correspondences for PnPRansac
#     if len(valid_points1) < 4:
#         return None, None

#     valid_points1 = np.array(valid_points1).astype(np.float64)/resize_factor

#     valid_points1[:,0] += y_offset
#     valid_points1[:,1] += x_offset

#     valid_points1[:,[0,1]] = valid_points1[:,[1,0]]

#     try:
#         retval, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(points2_3D).astype(np.float64), valid_points1, cam_K,
#                                                         distCoeffs=None, iterationsCount=100, reprojectionError=8.0)
    
#     except:
#         print("What the Fuck")
#         return None, None
    
#     R_est, _ = cv2.Rodrigues(rvec)
#     t_est = np.squeeze(tvec) * scale_factor

#     return R_est, t_est


    







    

    





