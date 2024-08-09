"""
Author: Swati Shirke
Date: 31 January, 2024
This code is written for camera calibration.
Reference is 
"""
import os
import cv2
import numpy as np
import scipy.optimize

def loadImages(folder_name, image_files):
	print("Loading images from ", folder_name)
	images = []
	if image_files is None:
		image_files = os.listdir(folder_name)
	for file in image_files:
		#print("Loading image ", file, " from ", folder_name)
		image_path = folder_name + "/" + file		
        image = cv2.imread(image_path)   
        
		image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		if image is not None:
			images.append(image)
		else:
			print("Error in loading image ", image)

	return images

def get_corner_list(image_list, pattern_size):
    """
    image_list: list of input images
    pattern_size: the size of checkerboard pattern

    returns
    corners_list: detetced checkerboard corners list for all images

    """
    corners_list = []
    for img in image_list:
        ret, corners = cv2.findChessboardCorners(img, pattern_size, None)
        if ret:
            corners_list.append(corners)
    
    return np.array(corners_list)

def get_world_corners(pattern_size, checkbox_size):
    """ 
    get world's corners corresponding to checkerboard corners
    pattern_size: the size of checkerboard pattern
    checkbox_size: size of one square of checkerboard

    returns:
    coordinates_list: list of real world corners


    """
    x_values = np.arange(0, pattern_size[0] * checkbox_size, checkbox_size)
    y_values = np.arange(0, pattern_size[1] * checkbox_size, checkbox_size)
    
    x,y = np.meshgrid( x_values, y_values)    
    coordinates_list = np.vstack([[x.flatten(), y.flatten(), np.zeros(len(x.flatten()))]]).T
        
    return coordinates_list

def get_homography_list(corners_list, world_corners_list):
    """
    corners_list: input corners list detected from images
    world_corners_list: real world corner list

    returns:
    H_list: list of Homography matrix between image and real world corners

    """
    H_list = []
    for i in range(len(corners_list)):
        c_list  = corners_list[i]
        H,_ = cv2.findHomography(world_corners_list, c_list, cv2.RANSAC, 5.0)
        H_list.append(H)
        
    return H_list

def get_v_equ_matrix(homograhpy_list):

    v_mat = []
    for H in homograhpy_list:
        v_12 = get_v_equ(H,1,2)
        v_11 = get_v_equ(H,1,1)
        v_22 = get_v_equ(H,2,2)
        v_mat.append(v_12)
        v_mat.append(v_11 - v_22)
    v_mat = np.array(v_mat)
    
    return v_mat   


def get_v_equ(H,i,j):
    i = i-1
    j = j-1
    v =  [ H[0][i]* H[0][j], 
        H[0][i]* H[1][j] + H[1][i]* H[0][j],
        H[1][i]* H[1][j],
        H[2][i]* H[0][j] + H[0][i]* H[2][j],
        H[2][i]* H[1][j] + H[1][i]* H[2][j],
        H[2][i]* H[2][j] ]  
    return np.array(v)

def compute_intrinsic_matrix(v_mat):
    """
    calculates intrinsic matrix from the given v_mat list using eigen values
    """
    U, S, V = np.linalg.svd(v_mat)    
    b = V.T[:,-1]    
    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]   
    
    v_0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12)
    lambda_ = (B33 - (B13 * B13 + v_0 * (B12 * B13 - B11 * B23)) / B11 )
    alpha = (lambda_ / B11)**0.5
    beta =  (lambda_ * B11 / (B11 * B22 - B12 * B12))** 0.5
    gamma = - B12 * alpha**2 * beta /lambda_ 
    u_0 = gamma * v_0 / beta - B13* alpha**2 / lambda_ 
    A_mat = create_A_mat(alpha, beta, gamma, u_0, v_0)
    
    return A_mat
    

def create_A_mat(alpha, beta, gamma, u_0, v_0):
    """
    This creates A matrix from given parameters as per the paper.
    """
    
    row_1 = [alpha, gamma, u_0]
    row_2 = [0, beta, v_0]
    row_3 = [0, 0, 1]
    A = np.vstack((row_1,row_2,row_3))    
    return  A

def compute_extrinsic_matrix(homograhpy_list, K):
    """
    Calculates extrinisic parameter matrix 
    """
    R = []
    for H in homograhpy_list:
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]
        lambda_ext = 1 / np.linalg.norm(np.dot(np.linalg.pinv(K), h1),2)
        r1 = lambda_ext * np.dot(np.linalg.pinv(K), h1)
        r2 = lambda_ext * np.dot(np.linalg.pinv(K), h2)
        r3 = np.cross(r1, r2)
        t = lambda_ext * np.dot(np.linalg.pinv(K), h3)
        rt = np.vstack((r1, r2, t)).T
        R.append(rt)

    return R

def calculate_loss(K, R, corners_list, world_corners_list):
    """
    this function calculates loss for all 13 images using non-linear optimization method
    """
    alpha, gamma, beta, u_0, v_0, k1, k2 = K 

    K = np.array([[alpha , gamma, u_0],
                  [0 , beta , v_0],
                  [0,0,1]])    
    
    error_list = []
    for i,corners in enumerate(corners_list):     ##13 cornerlists we have, because of 13 images 
        cumm_error = 0 
        for j, w_corners in enumerate(world_corners_list):   ##only one world list we have 
            T = np.dot(K, R[i])
            ## x and y in camera frames
            world_point = np.array([ w_corners[0], w_corners[1], 1])            
            c_frame_point = np.dot(R[i], world_point.T)
            c_frame_point /= c_frame_point[2]
            x, y  = c_frame_point[0],c_frame_point[1]
            
            ##point projected on image plane 
            projected_point = np.dot(T, world_point.T)
            projected_point /= projected_point[2]
            u, v = projected_point[0], projected_point[1]
            
            ##point corrected considering lense distortion k1,k2
            u_hat =  u + (u-u_0) *(k1 * (x**2 + y**2) + k2 * (x**2 + y**2)**2 )
            v_hat =  v + (v-v_0) * (k1 * (x**2 + y**2) + k2 * (x**2 + y**2)**2 )
            corrected_point = np.array([u_hat, v_hat, 1]) 
            
            img_corner  = np.array([corners[j][0][0], corners[j][0][1], 1 ])
            error = np.linalg.norm(img_corner - corrected_point, 2)
            cumm_error += error  
        
        error_list.append(cumm_error / 54.0)

    return np.array(error_list)  

def optimize_K(K, R, corners_list, world_corners_list):
    alpha = K[0][0]
    beta = K[1][1]
    gamma = K [0][1]
    u_0 = K[0][2]
    v_0 = K[1][2]
    k1 = 0
    k2 = 0
    optimized = scipy.optimize.least_squares(fun=calculate_loss , x0 = [alpha, gamma, beta, u_0, v_0, k1, k2], method = 'lm', args=(R, corners_list, world_corners_list))
    [alpha_u, gamma_u, beta_u, u0_u, v0_u ,k1_u, k2_u] = optimized.x
    updated_K = np.array([[alpha_u , gamma_u, u0_u],
                  [0, beta_u, v0_u],
                  [0, 0, 1]])

    return updated_K, k1_u, k2_u              

def compute_projetced_points(K, R, k1, k2,  corners_list, world_corners_list):
    """
    this function calculates projetced points for all 13 images 
    """
    u_0 = K[0][2]
    v_0 = K[1][2] 
    
    proj_error_list = [] 
    proj_point_list = []
    for i,corners in enumerate(corners_list):     ##13 cornerlists we have, because of 13 images 
        cumm_error = 0 
        proj_corner_list = []
        for j, w_corners in enumerate(world_corners_list):   ##only one world list we have 
            T = np.dot(K, R[i])
            ## x and y in camera frames
            world_point = np.array([ w_corners[0], w_corners[1], 1])            
            c_frame_point = np.dot(R[i], world_point.T)
            c_frame_point /= c_frame_point[2]
            x, y  = c_frame_point[0],c_frame_point[1]
            
            ##point projected on image plane 
            projected_point = np.dot(T, world_point.T)
            projected_point /= projected_point[2]
            u, v = projected_point[0], projected_point[1]
            
            ##point corrected considering lense distortion k1,k2
            u_hat =  u + (u-u_0) *(k1 * (x**2 + y**2) + k2 * (x**2 + y**2)**2 )
            v_hat =  v + (v-v_0) * (k1 * (x**2 + y**2) + k2 * (x**2 + y**2)**2 )
            corrected_point = np.array([u_hat, v_hat, 1]) 
            
            ##projetced points 
            img_corner  = np.array([corners[j][0][0], corners[j][0][1], 1 ])
            error = np.linalg.norm(img_corner - corrected_point, 2)
            cumm_error += error  
            proj_corner_list.append(corrected_point)

        
        proj_point_list.append(proj_corner_list)
        proj_error_list.append(cumm_error)

    avg_error = np.sum(np.array(proj_error_list)) / (len(corners_list) * world_corners_list.shape[0])

    return proj_point_list, avg_error    

def main():
    folder_name = "./Data"
    file_names = os.listdir(folder_name)
    file_path = "./Data"
    image_list = loadImages(folder_name, file_names)        
    pattern_size = (9,6)
    checkbox_size = 21.5
    corners_list = get_corner_list(image_list, pattern_size)   

    ##find image corner list - list of 13 lists of corners of 13 images and 1 corner list of real world point        
    world_corners_list = get_world_corners(pattern_size, checkbox_size)      # 13, * 54 * 2
    homograhpy_list = get_homography_list(corners_list, world_corners_list)  # 13 * 3 * 3     
    v_mat = get_v_equ_matrix(homograhpy_list)                                # (13 * 2) * 6 
    K = compute_intrinsic_matrix(v_mat)                                      # 3 * 3
    R_list = compute_extrinsic_matrix(homograhpy_list, K)                    # 13 * 3 * 3
    print(K)

    #print(np.shape(R_list))
    ##optimize and find updated intrinsic and extrinsic matrices
    updated_K, k1_u, k2_u = optimize_K(K, R_list, corners_list, world_corners_list)
    updated_R = compute_extrinsic_matrix(homograhpy_list, updated_K)
    updated_dist = np.array([k1_u, k2_u,0,0], dtype=np.float64)
    print(updated_K)
    #print(updated_R)
    print([k1_u,k2_u])
    

    ##find projected points again with new K and R
    proj_point_list, avg_error = compute_projetced_points(updated_K, updated_R, k1_u, k2_u, corners_list, world_corners_list)
    print(avg_error)
    for i,image_points in enumerate(proj_point_list):
        image = cv2.undistort(image_list[i], updated_K , updated_dist)
        for point in image_points:
            image = cv2.circle(image, (int(point[0]),int(point[1])), 5, (0, 0, 255), 5)
        # cv2.imshow('frame', image)
        filename =file_path + "/"+ str(i) + "reprojected_image.png"
        cv2.imwrite(filename, image)


        
    


if __name__=="__main__":
    main()
