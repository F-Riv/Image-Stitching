import cv2
import matplotlib.pyplot as plt
import numpy as np




def clean_global_vars():
    global color_image_1, color_image_2, gray_image_1, gray_image_2
    color_image_1, color_image_2 = None, None
    gray_image_1, gray_image_2 = None, None



# Assign array images to global variables
def assign_img_from_array(left_image, right_image):
    clean_global_vars()
    color_image_1 = left_image
    color_image_2 = right_image
    gray_image_1 =  cv2.cvtColor(color_image_1, cv2.COLOR_RGB2GRAY)
    gray_image_2 =  cv2.cvtColor(color_image_2, cv2.COLOR_RGB2GRAY)
    return color_image_1, color_image_2, gray_image_1, gray_image_2

    

# Read images from files and assign them to global variables assignment images are used by default and convert to gray scale
def read_img_from_files(img_file_1="images/image1.png", img_file_2="images/image2.png", plot="on"): 
    clean_global_vars()
    color_image_1 = cv2.imread(img_file_1)
    color_image_2 = cv2.imread(img_file_2)
    gray_image_1 =  cv2.cvtColor(color_image_1, cv2.COLOR_RGB2GRAY)
    gray_image_2 =  cv2.cvtColor(color_image_2, cv2.COLOR_RGB2GRAY)
    return color_image_1, color_image_2, gray_image_1, gray_image_2

def convert_to_gray():
    
    gray_image_1 = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2GRAY)
    gray_image_2 = cv2.cvtColor(color_image_2, cv2.COLOR_BGR2GRAY)    
    return gray_image_1, gray_image_2
    



    
    
    

    
    # Split an image in horizontal in two overlapping parts if you have a whole image
def split_image(img_path): 
    # read img from file
    image = cv2.imread(img_path)
    split_percentage = 0.5

    width = image.shape[1]
    split_position_left = int(width * (split_percentage+0.15))
    split_position_right= int(width * (split_percentage-0.15))


    left_image = image[:, :split_position_left]
    right_image = image[:, split_position_right:]
    assign_img_from_array(left_image, right_image)

    global color_image_1, color_image_2
    color_image_1, color_image_2 = left_image, right_image  

    return left_image, right_image  


# Convert cv2 points to cordinates
def keypoints_to_points(keypoints):
    points = np.zeros((len(keypoints), 2), dtype=np.float32)
    for i, kp in enumerate(keypoints):
        points[i] = kp.pt
    return points


# Get source points and destination points for RANSAC
def get_src_dst_points(matches_concat, kp_1, kp_2):
    src_match =   [p[0] for p in matches_concat ]
    dst_match =   [p[1] for p in matches_concat ]
    src = []
    dst = []
    for p in src_match: 
        src.append(kp_1[p])

    for q in dst_match:    
        dst.append(kp_2[q])

    source_points = keypoints_to_points(src)
    destination_points = keypoints_to_points(dst)
    return source_points, destination_points


# Create a dictionary from a single list of values 
def create_parameters_dict_from_list(distance_type, top_n_values, ransac_iterations, ransac_inlier_threshold, peak_min_distance, patch_size, k):
    parameters = {
        'Distance type': [distance_type],
        'Top n values': [top_n_values],
        'RANSAC iterations': [ransac_iterations],
        'RANSAC inlier threshold': [ransac_inlier_threshold],
        'Peak min distance': [peak_min_distance],
        'SIFT Patch size': [patch_size],
        'k Harris': [k]
    }
    return parameters

