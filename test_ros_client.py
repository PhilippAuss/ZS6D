from actionlib import SimpleActionClient
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pose_utils.img_utils as img_utils
import pose_utils.vis_utils as vis_utils
import rospy
import ros_numpy
import transforms3d as tf3d

from sensor_msgs.msg import Image
from robokudo_msgs.msg import GenericImgProcAnnotatorAction, GenericImgProcAnnotatorGoal, GenericImgProcAnnotatorResult


def rle_to_ros_image(mask):
    mask_binary = img_utils.rle_to_mask(mask)

    plt.imshow(mask_binary.astype(bool))
    plt.savefig("ros_mask.png")

    mask_ros = ros_numpy.msgify(
        Image,
        mask_binary.astype(np.uint8),
        encoding='8UC1')

    mask_ros.is_bigendian = 1 if mask_ros.is_bigendian else 0

    return mask_ros


if __name__ == "__main__":
    idx = 0
    rospy.init_node('dummy_client')

    with open("gts/test_gts/ycbv_bop_test_gt_sam.json") as fp:
        gts = json.load(fp)

    img_gt = gts["000048_1"]
    img_gt = gts["000052_543"]
    rgb = cv2.imread("./test/000543.png")[...,::-1]
    cam_K = np.array(img_gt[idx]['cam_K']).reshape((3,3))

    plt.imshow(rgb)
    plt.savefig("ros_input.png")

    # Create action client
    client = SimpleActionClient('/pose_estimator/get_poses', GenericImgProcAnnotatorAction)
    client.wait_for_server()

    # Create action goal
    goal = GenericImgProcAnnotatorGoal()
    goal.mask_detections = [rle_to_ros_image(img_gt[idx]['mask_sam'])]

    goal.class_names = [str(img_gt[idx]['obj_id'])]
    goal.rgb = ros_numpy.msgify(Image, rgb, encoding='8UC3')
    goal.rgb.is_bigendian = 1 if goal.rgb.is_bigendian else 0
    # goal.depth =
    client.send_goal(goal)

    # Receive results
    client.wait_for_result()
    result = client.get_result()

    out_img = vis_utils.draw_3D_bbox_on_image(
        np.array(rgb),
        tf3d.quaternions.quat2mat(
            ros_numpy.numpify(result.pose_results[0].orientation)),
        ros_numpy.numpify(result.pose_results[0].position)*1000,
        cam_K,
        img_gt[idx]['model_info'],
        factor=1.0)

    plt.imshow(out_img); plt.savefig("debug_imgs/ros_result.png")

    print(f"Results:\n===")
    print("Pred:", result.pose_results[0].position)
    print("GT:  ", np.array(img_gt[idx]['cam_t_m2c'])/1000)

    print("Pred:", result.pose_results[0].orientation)
    print("GT:  ", tf3d.quaternions.mat2quat(np.array(img_gt[idx]['cam_R_m2c'])))