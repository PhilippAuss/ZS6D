import os.path as osp
import sys

from zs6d import ZS6D
import os
import argparse
import time

import numpy as np

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
import actionlib
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction
import json
import ros_numpy
import transforms3d as tf3d
import matplotlib.pyplot as plt
import pose_utils.vis_utils as vis_utils


class ZS6D_ROS:
    def __init__(self, config_file):
            print(f"Using config file: {config_file}")
            with open(os.path.join(config_file), 'r') as f:
                config = json.load(f)

            self.object_name_mapping = config["object_mapping"]
            self.intrinsics = np.asarray(
                 config['cam_K']
                ).reshape((3,3))
            print(f"Using intrinsics: {self.intrinsics}")

            rospy.loginfo("Initializing zs6d")
            self.zs6d_predictor = ZS6D(
                os.path.join(config['templates_gt_path']),
                os.path.join(config['norm_factor_path']),
                model_type='dino_vits8',
                stride=4,
                subset_templates=config['template_subset'],
                max_crop_size=80)

            rospy.init_node("zs6d_estimation")
            self.server = actionlib.SimpleActionServer('/pose_estimator/zs6d',
                                                        GenericImgProcAnnotatorAction,
                                                        execute_cb=self.estimate_pose,
                                                        auto_start=False)
            self.server.start()
            print("Pose Estimation with ZS6D is ready.")

    """
    When using the robokudo_msgs, as the callback function for the action server
    """
    def estimate_pose(self, req):
        print("request detection...")
        start_time = time.time()

        # === IN ===
        # --- rgb
        # bb_detections = req.bb_detections
        mask_detections = req.mask_detections
        class_names = req.class_names
        # description = req.description
        rgb = req.rgb
        depth = req.depth

        width, height = rgb.width, rgb.height
        # assert width == 640 and height == 480

        image = ros_numpy.numpify(rgb)

        try:
            depth_img = ros_numpy.numpify(depth)
        except Exception as e:
             rospy.logwarn("Missing depth image in the goal.")

        print("RGB", image.shape, image.dtype)


        mask_detections = [ros_numpy.numpify(mask_img).astype(np.uint8)
                            for mask_img in req.mask_detections]

        print("mask", mask_detections[0].shape, mask_detections[0].dtype)


        valid_class_names = []
        pose_results = []
        # class_confidences = []
        for name, mask in zip(class_names, mask_detections):
            R_est, t_est = self.zs6d_predictor.get_pose(
                image,
                self.object_name_mapping[name],
                mask,
                self.intrinsics)

            quat = tf3d.quaternions.mat2quat(R_est)
            pose = Pose()
            t_est_meter = t_est/1000
            pose.position.x = t_est_meter[0]
            pose.position.y = t_est_meter[1]
            pose.position.z = t_est_meter[2]
            pose.orientation.w = quat[0]
            pose.orientation.x = quat[1]
            pose.orientation.y = quat[2]
            pose.orientation.z = quat[3]
            pose_results.append(pose)
            valid_class_names.append(name)

        response = GenericImgProcAnnotatorResult()
        response.pose_results = pose_results
        response.class_names = valid_class_names
        # response.class_confidences = class_confidences

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time:', elapsed_time, 'seconds')
        self.server.set_succeeded(response)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="./zs6d_configs/bop_eval_configs/cfg_ros_ycbv_inference_bop.json")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    ZS6D_ROS(**vars(opt))
    rospy.spin()
