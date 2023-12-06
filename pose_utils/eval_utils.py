import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R


def opencv2opengl(cam_matrix_world):
    """
    Change coordinate system from OpenCV to OpenGL or from OpenGL to OpenCV
    """
    rot180x = R.from_euler('x', 180, degrees=True).as_matrix()
    rotation = cam_matrix_world[:3, :3]
    translation = cam_matrix_world[:3, 3]
    output = np.copy(cam_matrix_world)
    output[:3,:3] = np.asarray(rot180x) @ rotation
    output[:3, 3] = np.asarray(rot180x) @ translation
    # output[:3, :3] = np.asarray(Matrix(rot180x) @ Matrix(rotation).to_3x3())
    # output[:3, 3] = np.asarray(Matrix(rot180x) @ Vector(translation))
    return output

def calculate_score(R_est, R_gt, obj_id, id_symmetry=0):
    # Returns: err, acc

    R_est = torch.tensor(R_est)
    R_gt = torch.tensor(R_gt)
    id_obj = torch.tensor(obj_id)

    output = _calculate_score(R_est, R_gt, id_symmetry, id_obj, id_obj)


    return output[0][int(obj_id)].item(), output[1][int(obj_id)].item()




def _calculate_score(pred_location, gt_location, id_symmetry, id_obj, pred_id_obj):
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