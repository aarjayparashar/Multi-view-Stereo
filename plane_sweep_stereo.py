import numpy as np
import cv2
import pdb
import tqdm
import math

EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    points = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2, 2, 3)



    """ YOUR CODE HERE
    """
    points_3d_normalized = np.tensordot(np.linalg.inv(K), points, axes=[1, 2])
    points_3d_normalized = np.transpose(points_3d_normalized, (1, 2, 0))
    xyz_cam = np.stack((depth * points_3d_normalized[:, :, 0], depth * points_3d_normalized[:, :, 1],
                        depth * points_3d_normalized[:, :, 2]), axis=2)
    R = Rt[:, :3]
    T = Rt[:, 3:]
    xyz_cam_temp = xyz_cam[:, :, :, None] - T
    xyz_cam_temp = xyz_cam_temp.squeeze(3)
    xyz_world = np.tensordot(R.T, xyz_cam_temp, axes=[1, 2])
    points = np.transpose(xyz_world, (1, 2, 0))
    """ END YOUR CODE
    """
    return points

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    R = Rt[:, :3]
    H = points.shape[0]
    W = points.shape[1]

    T = Rt[:, 3:].reshape((3,1))
    # print(f"T shape: {T.shape}")
    temp_points = np.ones((2,2,2))
    for i in range(H):
        for j in range(W):
            temp_point = K @ (R @ points[i, j].reshape(3,1) + T)
            # print(f"temp_point shape {temp_point.shape}")
            temp_points[i, j] = temp_point[:2, 0]/temp_point[2, 0]

    # points = points[:, :, :2]
    points = temp_points
    """ END YOUR CODE
    """
    return points

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]

    """ YOUR CODE HERE
    """

    """
    1. From the neighbouring frame and the reference frame, back project 4 corner points onto the imaginary frame at the
     given depth.
    2. The find the homography matrix between the set of points found in step 1 on the imaginary depth plane.
    3. Now, transform the points from the poi"""

    points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
    points = points.T
    backproject_ref_corners = backproject_fn(K_ref, width, height, depth, Rt_ref)
    project_ref_corners_to_neighbor = project_fn(K_neighbor, Rt_neighbor, backproject_ref_corners)
    a = np.zeros((2,1))

    for i in range(project_ref_corners_to_neighbor.shape[0]):
        for j in range(project_ref_corners_to_neighbor.shape[1]):
            a = np.hstack((a, project_ref_corners_to_neighbor[i,j,:].reshape((2,1))))
    a = a[:, 1:]
    H_ref_to_neighbor = cv2.findHomography(points.T, a.T, cv2.RANSAC)
    H_ref_to_neighbor = H_ref_to_neighbor[0]
    H_neighbor_to_ref = np.linalg.inv(H_ref_to_neighbor)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, H_neighbor_to_ref, (neighbor_rgb.shape[1],neighbor_rgb.shape[0]))

    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """
    M = src.shape[0]
    N = dst.shape[0]

    src_mean = np.mean(src, axis=2)
    src_zero_sum = src - src_mean[:, :, None, :]
    dst_mean = np.mean(dst, axis=2)
    dst_zero_sum = dst - dst_mean[:, :, None, :]
    prod_zero_sum = src_zero_sum * dst_zero_sum
    src_std = np.std(src, axis=2) + EPS
    dst_std = np.std(dst, axis=2) + EPS
    num = np.sum(prod_zero_sum, axis=2)
    denom = src_std * dst_std
    zncc_rgb = num / denom
    zncc = np.sum(zncc_rgb, axis=2)

    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))
    points = np.stack((_u, _v, np.ones(_u.shape)), axis=2)
    """ YOUR CODE HERE
    """
    points_3d_normalized = np.tensordot(np.linalg.inv(K), points, axes=[1, 2])
    points_3d_normalized = np.transpose(points_3d_normalized, (1, 2, 0))
    xyz_cam = np.stack((dep_map * points_3d_normalized[:, :, 0], dep_map * points_3d_normalized[:, :, 1],
                        dep_map * points_3d_normalized[:, :, 2]), axis=2)
    """ END YOUR CODE
    """
    return xyz_cam

