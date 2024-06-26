import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import math
import cv2
import open3d as o3d


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2) # Corners of the source (before)
    # print("source corners shape", corners_bef.shape)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)   # Corners of the destination (after)
    # print("destination corners shape", corners_aft)

    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    # print(f"umin: {u_min}, u_max: {u_max}, v_min: {v_min}, v_max: {v_max}")
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]
    # print("H shape", h)
    # print("W shape", w)
    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    print(f"Left view: {ui_min} {ui_max} { vi_min} {vi_max}")

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx" # This assumes that the X-coord of the centres of both the image planes are same
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    """Student Code Starts"""
    X_i = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    Y_i = np.float32([[0, 0], [ui_max - ui_min, 0], [ui_max - ui_min, vi_max - vi_min], [0, vi_max - vi_min]])
    

    X_j = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    Y_j = np.float32([[0, 0], [uj_max - uj_min, 0], [uj_max - uj_min, vj_max - vj_min], [0, vj_max - vj_min]])
    
    H_i = K_i_corr @ R_irect @np.linalg.inv(K_i)
    H_j = K_j_corr @ R_jrect @np.linalg.inv(K_j)


    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, dsize=(w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, dsize=(w_max, h_max))
    """Student Code Ends"""

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_wi, R_wj : [3,3]
    T_wi, T_wj : [3,1]
        p_i = R_wi @ p_w + T_wi
        p_j = R_wj @ p_w + T_wj
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ji @ p_j + T_ji, B is the baseline
    """

    """Student Code Starts"""
    R_wj_inv = np.linalg.inv(R_wj)
    R_ji = R_wi @ R_wj_inv
    T_ji = T_wi - R_ji @ T_wj
    B = np.linalg.norm(T_ji)
    """Student Code Ends"""

    return R_ji, T_ji, B


def compute_rectification_R(T_ji):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ji : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ji.squeeze(-1) / (T_ji.squeeze(-1)[1] + EPS)

    """Student Code Starts"""

    r1 = (np.array([[T_ji[1, 0]], [-T_ji[0, 0]], [0]])/np.linalg.norm(T_ji[:2, :])).reshape(-1, )
    r2 = (T_ji/np.linalg.norm(T_ji)).reshape(-1, )
    r3 = np.cross(r1, r2)
    R_irect = np.vstack((r1, r2, r3))

    """Student Code Ends"""

    return R_irect


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """

    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""

    M = src.shape[0]
    N = dst.shape[0]
    temp = (src[:,None,:,:] - dst[None, :, :, :])**2
    ssd = np.sum(np.sum(temp, axis=3), axis=2)
    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    M = src.shape[0]
    N = dst.shape[0]
    sad = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            sad[i, j] = np.sum(abs(src[i, :, :] - dst[j, :, :]))
    """Student Code Ends"""

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    M = src.shape[0]
    N = dst.shape[0]
    zncc = np.zeros((M, N))
    for i in tqdm(range(M)):
        for j in range(N):
            n = src.shape[1]
            w_i_src = src[i, :, :] # includes all channels r-g-b
            w_i_dst = dst[j, :, :] # includes all channels r-g-b

            w_i_src_mean = np.mean(w_i_src, axis=0)
            w_i_dst_mean = np.mean(w_i_dst, axis=0)
            num = np.sum((w_i_src - w_i_src_mean)*(w_i_dst - w_i_dst_mean), axis=0)
            # print(w_i_src)
            std_i_src = np.sum((w_i_src - w_i_src_mean)**2, axis=0)/n
            for k in range(3):
                std_i_src[k] = math.sqrt(std_i_src[k])
            std_i_dst = np.sum((w_i_dst - w_i_dst_mean) ** 2, axis=0) / n
            for k in range(3):
                std_i_dst[k] = math.sqrt(std_i_dst[k])
            denom = std_i_src * std_i_dst + EPS
            zncc[i, j] = np.sum(num/denom)
    """Student Code Ends"""

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    """Student Code Starts"""
    H = image.shape[0]
    W = image.shape[1]
    buffer_patch = np.zeros((H, W, k_size**2, image.shape[2]), dtype='float64')
    for x in range(H):
        for y in range(W):
            count = 0
            for i in range(-int(k_size / 2.0), int(k_size / 2.0) + 1):
                idx = i + x
                for j in range(-int(k_size / 2.0), int(k_size / 2.0) + 1):
                    idy = j + y
                    if idy < 0 or idy >= W or 0 > idx or idx >= H:
                        continue
                    else:
                        buffer_patch[x, y, count, :] = image[idx, idy, :]
                    count += 1
    patch_buffer = buffer_patch.copy()

    """Student Code Ends"""
    return patch_buffer  # H,W,K**2,3

# def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel):
#     """Compute the disparity map from two rectified view

#     Parameters
#     ----------
#     rgb_i,rgb_j : [H,W,3]
#     d0 : see the hand out, the bias term of the disparty caused by different K matrix
#     k_size : int, optional
#         The patch size, by default 3
#     kernel_func : function, optional
#         the kernel used to compute the patch similarity, by default ssd_kernel

#     Returns
#     -------
#     disp_map: [H,W], dtype=np.float64
#         The disparity map, the disparity is defined in the handout as d0 + vL - vR

#     lr_consistency_mask: [H,W], dtype=np.float64
#         For each pixel, 1.0 if LR consistent, otherwise 0.0
#     """

#     """Student Code Starts"""
#     H = rgb_i.shape[0]
#     W = rgb_i.shape[1]
    
#     patches_i = img2patch_func(rgb_i.astype(float) / 255.0, k_size)  # [h,w,k*k,3]
#     patches_j = img2patch_func(rgb_j.astype(float) / 255.0, k_size)  # [h,w,k*k,3]

#     vi_idx, vj_idx = np.arange(H), np.arange(H)
#     disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
   
#     valid_disp_mask = disp_candidates > 0.0

#     lr_consistency_mask = np.zeros((H, W), dtype=float)
#     disp_map = np.ones((H,W))

#     for u in tqdm(range(W)):
#         buf_i, buf_j = patches_i[:, u], patches_j[:, u]
#         value = ssd_kernel(buf_i, buf_j)
#         _upper = value.max() + 1.0
#         value[~valid_disp_mask] = _upper

#         best_matched_right_pixel = np.argmin(value, axis=1)
#         best_matched_left_pixel = np.argmin(value[:, best_matched_right_pixel], axis=0)
    
#         column_mask = best_matched_left_pixel == vi_idx
#         lr_consistency_mask[:, u] = column_mask
#         disp_map[vi_idx, u] = disp_candidates[vi_idx, best_matched_right_pixel]
#     """Student Code Ends"""

#     return disp_map, lr_consistency_mask


def compute_disparity_map(
    rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch
):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    
    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    """Student Code Starts"""
    
    h, w = rgb_i.shape[:2]

    patches_i = img2patch_func(rgb_i.astype(float) / 255.0, k_size)
    patches_j = img2patch_func(rgb_j.astype(float) / 255.0, k_size)

    vi_idx, vj_idx = np.arange(h), np.arange(h)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
    valid_disp_mask = disp_candidates > 0.0

    disp_map = np.zeros((h, w), dtype = np.float64)
    lr_consistency_mask = np.zeros((h, w), dtype = np.float64)

    for u in range(w):
        buf_i, buf_j = patches_i[:, u], patches_j[:, u]
        value = kernel_func(buf_i, buf_j)
        _upper = value.max() + 1.0
        value[~valid_disp_mask] = _upper
        
        for v in vi_idx:
            match_r = value[v].argmin()
            disp_map[v, u] = d0 + v - match_r
            if(value[:,match_r].argmin() == v):
                lr_consistency_mask[v, u] = np.float64(1.0)

    """Student Code Ends"""

    return disp_map, lr_consistency_mask

def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    """Student Code Starts"""
    dep_map = (1/disp_map)*K[1, 1]*B
    H = disp_map.shape[0]
    W = disp_map.shape[1]
    X = np.arange(H)
    Y = np.arange(W)
    XX,YY = np.meshgrid(Y,X)
    print(XX.shape)
    points_2d = np.stack((XX,YY, np.ones(XX.shape)), axis=2)
    print(f"points 2d (1st point): {points_2d[0, 0, :]}")
    
    points_3d_normalized = np.tensordot(np.linalg.inv(K), points_2d, axes=[1, 2])

    points_3d_normalized = np.transpose(points_3d_normalized, (1,2,0))
    print(
        f"cross check for tensor product -> 1st point: {points_2d[0, 0, :]}, 1st point in 3D: {np.linalg.inv(K) @ points_2d[0, 0, :].reshape(3, 1)}")
    print(f"1st point from tesordot: {points_3d_normalized[0, 0, :]}")
    xyz_cam = np.stack((dep_map*points_3d_normalized[:, :, 0], dep_map*points_3d_normalized[:, :, 1], dep_map*points_3d_normalized[:, :, 2]), axis=2)

    """Student Code Ends"""

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_wc,
    T_wc,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""
    pcl_world = R_wc.T @ (pcl_cam[:, :, None] - T_wc)
    pcl_world = pcl_world.squeeze(2)
    print(pcl_world.shape)
    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_wi, T_wi = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_wj, T_wj = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ji, T_ji, B = compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj)
    assert T_ji[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ji)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ji,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_wi,
        T_wc=R_irect @ T_wi,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
