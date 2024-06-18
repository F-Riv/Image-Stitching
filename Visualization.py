import cv2
import matplotlib.pyplot as plt
import numpy as np




# Scatter point on the images
def scatter_points(copy_image_1, copy_image_2, source_points, destination_points, inliers): 
    for i, (source, destination) in enumerate(zip(source_points, destination_points)):
        # if the point is an inlier it is colored in blue otherwise in red
        if  any(np.array_equal((source, destination), inl) for inl in inliers):
            color = (255, 150, 70)
            
        else: 
            color = (100, 100, 255)
        # Draw circles for source and destination points
        cv2.circle(copy_image_1, (int(source[0]), int(source[1])), 3, color, -1)  
        cv2.circle(copy_image_2, (int(destination[0]), int(destination[1])), 3, color, -1)  


# Visualize images with inliers and outliers according to RANSAC
def ransac_visualization(copy_image_1, copy_image_2, source_points, destination_points, inliers, parameters, ransac_stats, plot='on'):
    color_image_rgb = cv2.cvtColor(copy_image_1, cv2.COLOR_BGR2RGB)
    color_image_2_rgb = cv2.cvtColor(copy_image_2, cv2.COLOR_BGR2RGB)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    param_string = ', '.join([f'{k}: {v}' for k, v in parameters.items()])
    ransac_stats_string = ', '.join([f'{k}: {v}' for k, v in ransac_stats.items()])
    fig.suptitle('RANSAC inliers with \n ' + param_string +'\n Result \n' +ransac_stats_string)
    axs[0].imshow(color_image_rgb)
    axs[1].imshow(color_image_2_rgb)

    

    axs[0].axis('off')
    axs[1].axis('off')
    plt.show()  




# Scatter point on the images
def scatter_points_on_stiched_img(canvas, src_points, transf_dest_points, inliers): 
    for i, (src, dest_transf) in enumerate(zip(src_points, transf_dest_points)):
        cv2.circle(canvas, (int(src[0]), int(src[1])), 3, (255, 100, 100), -1)  
        cv2.circle(canvas, (int(dest_transf[0]), int(dest_transf[1])), 3, (255, 100, 100), -1)  

        
    

# Visualize the stiched image     
def stitched_image_visualization(canvas, parameters):
    plt.figure(figsize=(8, 6))
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    param_string = ', '.join([f'{k}: {v}' for k, v in parameters.items()])
    plt.title('Stitched image with: \n' + param_string)
    plt.imshow(canvas_rgb)
    plt.axis('off') 
    plt.show()
    