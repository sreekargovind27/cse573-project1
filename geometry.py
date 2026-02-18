import torch
from typing import Tuple

'''
Please do Not change or add any imports. 
'''

# --------------------------------------------------- task1 ----------------------------------------------------------

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> torch.Tensor:
    rot_xyz2XYZ = torch.eye(3, dtype=torch.float32)
    a = torch.deg2rad(torch.tensor(alpha, dtype=torch.float32))
    b = torch.deg2rad(torch.tensor(beta, dtype=torch.float32))
    g = torch.deg2rad(torch.tensor(gamma, dtype=torch.float32))


    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cg, sg = torch.cos(g), torch.sin(g)

    # rotation matrices
    Rx = torch.tensor([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=torch.float32)
    Ry = torch.tensor([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=torch.float32)
    Rz = torch.tensor([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=torch.float32)

    rot_xyz2XYZ = Rz @ Ry @ Rx

    return rot_xyz2XYZ

def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> torch.Tensor:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 tensor represents the rotation matrix from XYZ to xyz.
    '''
    rot_XYZ2xyz = torch.eye(3, dtype=torch.float32)
    rot_XYZ2xyz = findRot_xyz2XYZ(alpha, beta, gamma).T
    return rot_XYZ2xyz
"""
If your implementation requires implementing other functions.
Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()"
functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1:



#---------------------------------------------------------------------------------------------------------------------






# --------------------------------------------------- task2 ----------------------------------------------------------

# for the find_corner_img_coord function implementation:
# You are able to use opencv to detect corners in this function, resulting in numpy arrays,
# but you have to convert numpy arrays back to torch.Tensor form.
# (findChessboardCorners, cornerSubPix can be used to find the corners as the image coordinates)
# (drawChessboardCorners can be used to see if you find the true corners) you can see the true corners in the project pdf - figure 2
# Comment out the following three lines to import the useful functions you need:
import numpy as np
from cv2 import TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER,findChessboardCorners, cornerSubPix

def find_corner_img_coord(image: torch.Tensor) -> torch.Tensor:
    '''
    Args: 
        image: Input image of size 3xMxN.
        M is the height of the image.
        N is the width of the image.
        3 is the channel of the image.

    Return:
        A tensor of size 18x2 that represents the 18 checkerboard corners' pixel coordinates. 
        The pixel coordinate is as usually defined such that the top-left corner is (0, 0)
        and the bottom-right corner of the image is (N, M). 
    '''

    gray = image.squeeze(0).numpy().astype(np.uint8)

    # corner detection phase using findChessboardCorners
    res, corners = findChessboardCorners(gray, (7, 3))
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    all_pts = corners.reshape(-1, 2)

    cols = 7
    keep = [i for i in range(21) if i % cols != cols // 2]
    img_coord = torch.tensor(all_pts[keep], dtype=torch.float32)
    return img_coord


def find_corner_world_coord(img_coord: torch.Tensor) -> torch.Tensor:
    '''
    You can output the world coord manually or through some algorithms you design.
    Your output should be the same order with img_coord.

    Args: 
        img_coord: The image coordinate of the corners.
        Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.

    Return:
        A torch.Tensor of size 18x3 that represents the 18
        (21 detected points minus 3 points on the z axis look at the figure in the documentation carefully)... 
        ...checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image.
        The output results should be in milimeters.
    '''


    world_coord = torch.zeros(18, 3, dtype=torch.float32)

    # height
    z_vals = [40.0, 30.0, 20.0]

    # left facing y-values
    y_vals = [30.0, 20.0, 10.0]

    # right facing x-values
    x_vals = [10.0, 20.0, 30.0]

    for row in range(3):
        base = row * 6
        for col in range(3):
            world_coord[base + col] = torch.tensor([0, y_vals[col], z_vals[row]])
        for col in range(3):
            world_coord[base + 3 + col] = torch.tensor([x_vals[col], 0, z_vals[row]])
    return world_coord


def find_intrinsic(img_coord: torch.Tensor, world_coord: torch.Tensor) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 18 point to calculate the intrinsic parameters.

    Args: 
        img_coord: The image coordinate of the 18 corners. This is a 18x2 tensor.
        world_coord: The world coordinate of the 18 corners. This is a 18x3 tensor.

    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''
    
    M = _get_projection_matrix(img_coord, world_coord)

    # row vectors
    m1 = M[0, :3]
    m2 = M[1, :3]
    m3 = M[2, :3]

    # principal points
    cx = torch.dot(m1, m3).item()
    cy = torch.dot(m2, m3).item()

    # focal lengths
    fx = torch.sqrt(torch.dot(m1, m1) - cx * cx).item()
    fy = torch.sqrt(torch.dot(m2, m2) - cy * cy).item()
    return fx, fy, cx, cy


def find_extrinsic(img_coord: torch.Tensor, world_coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Use the image coordinates, world coordinates of the 18 point and the intrinsic
    parameters to calculate the extrinsic parameters.

    Args: 
        img_coord: The image coordinate of the 18 corners. This is a 18x2 tensor.
        world_coord: The world coordinate of the 18 corners. This is a 18x3 tensor.
    Returns:
        R: The rotation matrix of the extrinsic parameters.
            It is a 3x3 tensor.
        T: The translation matrix of the extrinsic parameters.
            It is a 1-dimensional tensor with length of 3.
    '''
    M = _get_projection_matrix(img_coord, world_coord)

    m1 = M[0, :3]
    m2 = M[1, :3]
    m3 = M[2, :3]

    cx = torch.dot(m1, m3).item()
    cy = torch.dot(m2, m3).item()

    fx = torch.sqrt(torch.dot(m1, m1) - cx * cx).item()
    fy = torch.sqrt(torch.dot(m2, m2) - cy * cy).item()

    # intrinsic matrix
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float64)

    # getting the extrinsic parameters by applying rotation and transalation
    K_inv = torch.linalg.inv(K)

    R = (K_inv @ M[:, :3]).float()  
    T = (K_inv @ M[:, 3]).float()
    return R, T


"""
If your implementation requires implementing other functions.
Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2:

def _get_projection_matrix(img_coord, world_coord):
    n = img_coord.shape[0]
    A = torch.zeros(2 * n, 12, dtype=torch.float64)
    for i in range(n):
        Xw, Yw, Zw = world_coord[i].double()
        x, y = img_coord[i].double()
        A[2*i]   = torch.tensor([Xw, Yw, Zw, 1, 0, 0, 0, 0, -x*Xw, -x*Yw, -x*Zw, -x])
        A[2*i+1] = torch.tensor([0, 0, 0, 0, Xw, Yw, Zw, 1, -y*Xw, -y*Yw, -y*Zw, -y])
    U, S, Vh = torch.linalg.svd(A)
    p = Vh[-1]
    M = p.reshape(3, 4)
    scale = torch.norm(M[2, :3])
    M = M / scale

    # checking if the world points are correct
    test_pt = torch.cat([world_coord[0].double(), torch.tensor([1.0], dtype=torch.float64)])
    s = M @ test_pt
    if s[2] < 0:
        M = -M
    return M
#---------------------------------------------------------------------------------------------------------------------
