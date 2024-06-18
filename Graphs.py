import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np




# NOT USED
def scatterplot_distance_keypoints(source_points, destination_points, residuals_inlier_avg,  residuals_tot, inliers):
    sns.set_theme()
    pastel_palette = sns.color_palette("pastel")
    '''             
    # Calculate the Euclidean distance between each pair of corresponding keypoints
    keypoints_image1_transformed =[]


    for kp in source_points:
        homogeneous_point = np.array([kp[0], kp[1], 1])
        transformed_point = np.dot(affine_model, homogeneous_point)
        keypoints_image1_transformed.append((transformed_point[0], transformed_point[1]))
        
    i = 0
    for idx, (kp1, kp2, kp3)  in enumerate(zip(keypoints_image1_transformed, destination_points, source_points)):



        distance = np.linalg.norm(kp1[:2] - kp2[:2])
        distance_2 = np.linalg.norm(kp2[:2] - kp3[:2])
'''
    for idx, (kp1, kp2)  in enumerate(zip(source_points, destination_points)): 

        # if the point is an inlier it is colored in blue otherwise in red
        if  any(np.array_equal((kp1, kp2), inl) for inl in inliers):
            color =  pastel_palette[0]
        else: 
            color = pastel_palette[3]

        plt.scatter(idx, residuals_tot[idx], marker='o', color=color, alpha=0.6)
        #plt.scatter(idx, distance_2, marker='o', color=color, alpha=1)
            
    plt.xlabel('Keypoint Index')
    plt.ylabel('Euclidean Distance')
    plt.title('Distances for Key-points')
    plt.grid(True)
    plt.show()
    sns.set()




# Histogram comparing the average residual euclidean distance between the inliers and the actual point 
def hist_nr_inliers(nr_inliers, curr_param_list, parameters):  
    plt.figure(figsize=(10,5)) 
    sns.set_theme()
    colors = sns.color_palette("pastel", n_colors=len(nr_inliers))
    plt.xticks(fontsize=7)
    plt.bar(curr_param_list, nr_inliers, color=colors)
    text = '\n'.join(parameters)
    plt.text(-0.10, -0.22, text, fontsize=7, transform=plt.gca().transAxes, ha='left', va='bottom')
    plt.ylabel('Nr. inliers')
    plt.tight_layout() 
    plt.title('Number of Inliers by Parameter Combination')


# Histogram comparing the number of inliers found in every experimet
def hist_nr_residuals(residuals, curr_param_list, parameters):  
    plt.figure(figsize=(10,5)) 
    sns.set_theme()
    colors = sns.color_palette("pastel", n_colors=len(residuals))
    plt.xticks(fontsize=7)
    plt.bar(curr_param_list, residuals, color=colors)
    text = '\n'.join(parameters)
    plt.text(-0.10, -0.22, text, fontsize=7, transform=plt.gca().transAxes, ha='left', va='bottom')
    plt.ylabel('Residuals normalized by threshold')
    plt.tight_layout() 
    plt.title('Average of inliers euclidean residuals  (norm) by Parameter Combination')




# Plot patch metrics 
def plot_patch_metrics(ssmi_scores, mse_scores, curr_param_list, nr_inliers, parameters):
    sns.set_theme()
    colors = sns.color_palette("pastel", n_colors=len(curr_param_list))
    width = 0.7/len(curr_param_list)
    alpha_list = []

    for nr in nr_inliers: 
        if nr < 20: 
            alpha_list.append(0.3)
        else: alpha_list.append(1)    


    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # Plotting SSMi scores
    for idx, (cp, ssmi, mse, alpha, color) in enumerate(zip(curr_param_list, ssmi_scores, mse_scores, alpha_list, colors)):
        axs[0].bar(cp, ssmi, color=color, alpha=alpha)
    axs[0].set_ylabel('SSMI Scores')
    axs[0].set_title('Avg SSMI Scores of Inlier Patches by Parameter Combination')

    # Plotting nornalized MSE
    for idx, (cp, ssmi, mse, alpha, color) in enumerate(zip(curr_param_list, ssmi_scores, mse_scores, alpha_list, colors)):
        axs[1].bar(cp, mse, color=color, alpha=alpha)
    axs[1].set_ylabel('Avg MSE')
    axs[1].set_title('Avg normalized MSE of Inlier Patches by Parameter Combination')
    plt.xticks(fontsize=7)
    text = '\n'.join(parameters)
    plt.text(-0.1, -0.55, text, fontsize=7, transform=plt.gca().transAxes, ha='left', va='bottom')

    plt.tight_layout()  
    plt.show()
