import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim




    # Plot ten patches per image 
def plot_patches(patches, patches_2):
        
        if len(patches) >= 10 and len(patches_2) >= 10:
            n = 10 

        else: n = np.min(len(patches) and len(patches_2))
    
        num_rows = 5
        num_cols = 4
        plt.figure(figsize=(12, 8))
        for i, (patch, patch_2) in enumerate(zip(patches[:n], patches_2[:n])):
            plt.subplot(num_cols,num_rows, i+1)
            plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            plt.title(f'Patch {i+1}')
            plt.axis('off')  
            plt.subplot(num_cols, num_rows, 10+i+1)
            plt.imshow(cv2.cvtColor(patch_2, cv2.COLOR_BGR2RGB))
            plt.title(f'Patch {i+1}')
            plt.axis('off')  

        plt.show()


# Genrate patches to compare areas in the image around the inliers and the points trasformed according to 

def generate_patches(image, image_2,  inliers_dest_points, transf_inl_dest_points,  size=10, plot='on'): 
    patches = []
    patches_2 = []
    kp_patches = []
    kp_patches_2 = []
    patch_size = 20

    if len(inliers_dest_points)== 0: 
        print("No inliers found, patches can not be created")

    for kp, kp_2 in zip(inliers_dest_points, transf_inl_dest_points):
        
        x,y = kp[0], kp[1]
        x_2, y_2 = kp_2[0], kp_2[1]
    
        # Cut patch from first image
        if (x >= size and x < image.shape[1] - size and y >= size and y < image.shape[0] - size):
            patch = image[int(y - patch_size // 2): int(y + patch_size // 2), int(x - patch_size // 2): int(x + patch_size // 2)]
            patches.append(patch)
            kp_patches.append((y,x))
        # Cut patch from second image 
        if (x_2 >= size and x_2 < image_2.shape[1] - size and y_2>= size and y_2 < image_2.shape[0] - size):
            patch_2 = image_2[int(y_2 - patch_size // 2): int(y_2 + patch_size // 2), int(x_2 - patch_size // 2): int(x_2 + patch_size // 2)]
            patches_2.append(patch_2)
            kp_patches_2.append((y_2,x_2))  

    if plot=='on':
        plot_patches(patches, patches_2)    

    return patches, patches_2
    
# Mean Squared Error function
def mse_norm(img_1, img_2):
    
    max_pixel_value = 255.0
    # MSE
    mse = np.mean((img_1 - img_2) ** 2)
    # Normalize for maximum pixel value distance in grayscale
    normalized_mse = mse / (max_pixel_value ** 2)
    return normalized_mse
    

# Compare lists of inlier patches using mse and ssmi
def compare_patches(list_patches_1, list_patches_2):

    # No patches to compute metrics on 
    if len(list_patches_1) < 0: 
        print("No patches were found, patches can not be compared")
        similarity_index = 0 
        mse = 1
        return similarity_index, mse

    list_mse = []
    list_ssim = []

    
    for patch_1, patch_2 in zip(list_patches_1, list_patches_2):
        # Convert patches to grayscale
        patch_gray_1 = cv2.cvtColor(patch_1, cv2.COLOR_BGR2GRAY)
        patch_gray_2 = cv2.cvtColor(patch_2, cv2.COLOR_BGR2GRAY)

        # Compute SSIM and MSE
        similarity_index = ssim(patch_gray_1, patch_gray_2)
        mse = mse_norm(patch_gray_1, patch_gray_2)
        
    
    
        list_ssim.append(similarity_index)
        list_mse.append(mse)

        

    # Average scores among all the inlier patches  
    avg_ssim = np.mean(list_ssim)  
    avg_mse = np.mean(list_mse)
    print("SSIM similarity between inlier patches:",avg_ssim)
    print("MSE distance between inlier patches:",avg_mse)
    return avg_ssim, avg_mse

