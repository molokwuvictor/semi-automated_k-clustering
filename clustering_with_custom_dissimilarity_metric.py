# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:00:14 2025

@author: Victor Molokwu, Ph.D.
"""
# ---------------------------
# Import data science libraries relevant to clustering
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids  # pip install scikit-learn-extra
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import matplotlib as mpl
import math

# Import auxiliary functions
import auxiliary_functions as cf

plt.rcParams["font.family"] =["Times New Roman"]
plt.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams['figure.dpi'] = 600   

# Two cases are considered -- where 
# The pressure transient response for the test sequence has been saved in an excel (.xlsx) data file. 
case=4
slope_info=True

if case==3:
    sheet_name='Sheet1'
    w_size=5
    n_clusters=3
else:
    sheet_name='Sheet2'
    w_size=5
    n_clusters=6

# ---------------------------
# Data Processing
# ---------------------------
df = pd.read_excel('datafile.xlsx', sheet_name=sheet_name)
X=df[['lndt','dp_dlndt','dp']]
lndt,dp_dlndt,dp=X.iloc[:,0],X.iloc[:,1],X.iloc[:,2]

lndp_dlndt=np.log(dp_dlndt)
lndp=np.log(dp)

grad=cf.bourdet_derivative(x=lndt,y=dp,L=0.1,transform_x=False,transform_y=False)
grad_2=cf.bourdet_derivative(x=lndt,y=lndp_dlndt,L=0.,transform_x=False,transform_y=False,)
grad_3=cf.bourdet_derivative(x=lndt,y=grad_2,L=0.,transform_x=False,transform_y=False,)

# Plot the data and the moving average
plt.figure(figsize=(6, 4))
plt.plot(lndt,lndp, label='Pressure_Difference', color='orange',linestyle='--')
plt.plot(lndt,lndp_dlndt, label='Software_Derivative', color='blue')
plt.plot(lndt,np.log(grad), label='Calc_Bourdet_Derivative', color='purple', linestyle='--')
plt.plot(lndt,grad_2, label='Second_Bourdet_Derivative', color='green', linestyle='--')
#plt.plot(lndt,grad_3, label='Third_Bourdet_Derivative', color='red', linestyle='--')

plt.title('Pressure Difference and Derivatives - Diagnostic Plots')
plt.xlabel('lnΔt')
plt.ylabel('ln(dΔp), ln(dΔp/dlnΔt), dln(dΔp/dlnΔt)/dlnΔt')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', linewidth=0.25, color='gray')
plt.show()

# Normalize the dataset using the minmax scaler
grad_2=cf.min_max_scaler(grad_2,limits=[-1,1])
lndt=cf.min_max_scaler(lndt,limits=[-1,1])
lndp_dlndt=cf.min_max_scaler(lndp_dlndt,limits=[-1,1])

# Define normalized inputs
x,y,z=np.array(lndt),np.array(lndp_dlndt),np.array(grad_2)

# ---------------------------
# Helper function to compute the minimal angle between two slopes (in degrees)
# ---------------------------
def angle_between_slopes(m1, m2):
    """
    Compute the minimal angle difference (in degrees) between two undirected lines 
    given their slopes. The result is in [0, 90].
    """
    theta1 = math.degrees(math.atan(m1))
    theta2 = math.degrees(math.atan(m2))
    diff = abs(theta1 - theta2)
    if diff > 90:
        diff = 180 - diff
    return diff

# ---------------------------
# Create windows from N-D data with regression for slope and median for center
# ---------------------------
def create_windows(data, window_size):
    """
    Splits the N-D data (n_points x n_dims) into non-overlapping windows.
    For each window, we store:
      - 'data': the raw points in the window,
      - 'median': the robust center (median over dimensions),
      - 'slope': regression slope computed from a linear fit on the first two dimensions (x vs. y),
      - 'index': the sequential window index.
    """
    windows = []
    n = data.shape[0]
    for i in range(0, n - window_size + 1, window_size):
    #for i in range(0, n, window_size):
        window_data = data[i:i+window_size]
        if window_data.shape[0] < 2:
            break
        median = np.median(window_data, axis=0)
        # Compute regression slope on the first two dimensions (x and y)
        if np.ptp(window_data[:, 0]) != 0:
            slope, _ = np.polyfit(window_data[:, 0], window_data[:, 1], 1)
        else:
            slope = 0.0
        windows.append({
            'data': window_data,
            'median': median,
            'slope': slope,
            'index': len(windows)  # sequential index
        })
    return windows

# ---------------------------
# Assign inverted-V block membership for a block of p consecutive windows
# ---------------------------
def assign_inverted_v_block(windows, p, early_time_index=2):
    """
    For each sliding block of p consecutive windows, check if any consecutive pair
    within the block satisfies the inverted-V condition (i.e., window[j]['slope'] > 0 
    and window[j+1]['slope'] < 0). If so, mark all windows in that block with 
    'inverted_block' = True.
    Also if found at an early_time_index, which is less than the threshold set by the user, 
    all windows below the index is marked with 'inverted_block'=True. 
    """
    n = len(windows)
    # Initialize block flag to False for all windows.
    for w in windows:
        w['inverted_block'] = False
    # Slide a window of length p across the windows.
    for i in range(n - p + 1):
        block = windows[i:i+p]
        found = False
        for j in range(p - 1):
            if block[j]['slope'] > 0 and block[j+1]['slope'] < 0:
                found = True
                break
        if found:
            if i>early_time_index:
                k=i
            else:
                k=0
            for j in range(k, i+p):
                windows[j]['inverted_block'] = True
    return windows

# ---------------------------
# Custom distance function 
# ---------------------------
def custom_distance(w1, w2, D_max, T_max, lambda_e=1.0, lambda_p=1.0, beta=0.5,
                    delta=1.0, threshold=1e-3, gamma_block=1.0):
    """
    Computes a custom distance between two windows as the sum of:
    
      E (Euclidean): normalized Euclidean distance between medians, weighted by lambda_e.
      
      P (Slope difference): minimal angle between slopes (normalized by 90°), weighted by lambda_p.
      
      T (Temporal penalty): normalized gap in window indices, weighted by beta.
      
      C (Concave meniscus check): For successive windows, if the estimated curvature R 
         (computed from slopes and x-spacing) is greater than 2*dx, subtract delta.
    
    Additionally, if two successive windows both have near-zero slopes (|slope| < threshold),
    the distance is overridden to 10% of the normalized Euclidean term.
    
    If both windows have the 'inverted_block' flag True, subtract gamma_block as a bonus.
    """
    index_diff = abs(w1['index'] - w2['index'])
    
    # Euclidean term
    median_dist = np.linalg.norm(w1['data'].flatten() - w2['data'].flatten())
    #median_dist = np.linalg.norm(w1['median'] - w2['median'])
    normalized_euclid = median_dist / D_max if D_max > 0 else median_dist
    
    # Override for near-zero slopes on successive windows
    if index_diff == 1 and abs(w1['slope']) < threshold and abs(w2['slope']) < threshold:
        return lambda_e * normalized_euclid * 0.1
    
    # Slope difference term
    angle_diff = angle_between_slopes(w1['slope'], w2['slope'])
    normalized_angle = angle_diff / 90.0
    
    # Temporal penalty for non-successive windows
    norm_temporal = (max(0, index_diff - 1) / T_max) if T_max > 0 else max(0, index_diff - 1)
    
    # Concave bonus: for successive windows, based on curvature.
    concave_bonus = 0.0
    if index_diff == 1:
        dx = abs(w1['median'][0] - w2['median'][0])
        if dx > 0:
            m1 = w1['slope']
            m2 = w2['slope']
            m_avg = (m1 + m2) / 2.0
            y_dd = (m2 - m1) / dx  # approximate second derivative
            if abs(y_dd) < 1e-6:
                R = float('inf')
            else:
                R = (1 + m_avg**2)**(1.5) / abs(y_dd)
            if R > 2 * dx:
                concave_bonus = -delta
    
    # Block bonus: if both windows are marked as in an inverted-V block, subtract bonus.
    block_bonus = 0.0
    if w1.get('inverted_block', False) and w2.get('inverted_block', False):
        block_bonus = -gamma_block
    
    E_term = lambda_e * normalized_euclid
    P_term = lambda_p * normalized_angle
    T_term = beta * norm_temporal
    
    total_distance = E_term + P_term + T_term + concave_bonus + block_bonus
    return max(total_distance, 0)

# ---------------------------
# Main function using standard k-medoids from scikit-learn-extra (PAM algorithm)
# ---------------------------
def main_kmedoids():
    # Generate synthetic N-D data (3D: x, y, z)
    np.random.seed(42)
    n_points = 300
    
    # Testing ...
    # x = np.linspace(0, 100, n_points)
    # y = np.piecewise(x,
    #                  [x < 33, (x >= 33) & (x < 66), x >= 66],
    #                  [lambda x: 0.5 * x + 5,
    #                   lambda x: -0.3 * x + 40,
    #                   lambda x: 0.8 * x - 20])
    # y += np.random.normal(0, 2, size=x.shape)
    # z = np.piecewise(x,
    #                  [x < 33, (x >= 33) & (x < 66), x >= 66],
    #                  [lambda x: -0.2 * x + 10,
    #                   lambda x: 0.4 * x - 5,
    #                   lambda x: -0.5 * x + 80])
    # z += np.random.normal(0, 2, size=x.shape)
    data = np.column_stack((x, y,))
    
    # Create windows from the data
    window_size = w_size  # points per window
    windows = create_windows(data, window_size)
    
    # Assign inverted-V block membership using a user-defined block size p.
    p = 4  
    windows = assign_inverted_v_block(windows, p)
    
    # Compute normalization factors:
    # D_max: maximum Euclidean distance between window medians
    D_max = 0
    for i in range(len(windows)):
        for j in range(i+1, len(windows)):
            dist_ij = np.linalg.norm(windows[i]['data'].flatten() - windows[j]['data'].flatten())
            # dist_ij = np.linalg.norm(windows[i]['median'] - windows[j]['median'])
            if dist_ij > D_max:
                D_max = dist_ij
    # T_max: maximum index difference (last window index)
    T_max = windows[-1]['index'] if windows else 1

    # Weight parameters and threshold
    lambda_e = 1.0
    lambda_p = 1.0
    beta = 0.5
    delta = 0.1
    threshold = 1.
    gamma_block = 1.0  # bonus weight for block membership

    if not slope_info:
        lambda_p=0.0
    
    # Create a closure for the distance function
    def dist_func(w1, w2):
        return custom_distance(w1, w2, D_max, T_max, lambda_e, lambda_p, beta,
                               delta, threshold=threshold, gamma_block=gamma_block)

    # Build the precomputed distance matrix
    n_windows = len(windows)
    distance_matrix = np.zeros((n_windows, n_windows))
    for i in range(n_windows):
        for j in range(i, n_windows):
            d = dist_func(windows[i], windows[j])
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d  # symmetry

    # Set the number of clusters and use standard KMedoids with algorithm 'pam'
    k = n_clusters
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', method='pam', random_state=42)
    kmedoids.fit(distance_matrix)
    labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_

    # Assign cluster labels for visualization
    for i, w in enumerate(windows):
        w['cluster'] = labels[i]
    
    # --- Visualization: 2D Projection (x vs. y) ---
    title='2D Projection (x, y) of N-D Data Clustering\n(using KMedoids (PAM) with Normalized & Weighted Metrics)'
    cf.plot_2d_clustering(data, windows, cf.reassign_clusters, title, medoid_indices=medoid_indices)

# ---------------------------
# Main function using K-Means
# ---------------------------
def main_kmeans():
    # Generate synthetic N-D data (example: 3D data: x, y, z)
    np.random.seed(42)

    data = np.column_stack((x, y,))
    
    # Create windows from the data using a specified window size
    window_size = w_size  # number of points per window
    windows = create_windows(data, window_size)
    
    # Build feature vectors for each window.
    # Here we use: [median_x, median_y, slope, index]
    features = []
    for w in windows:
        median = w['median']
        if slope_info:
            features.append([median[0], median[1], w['slope'], w['index']])
        else:
            features.append([median[0], median[1], w['index']])
    features = np.array(features)
    
    # Standardize the feature vectors so that each component contributes equally.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Run standard k-means on the scaled features.
    k = n_clusters  # number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    
    # Save cluster labels for each window.
    for i, w in enumerate(windows):
        w['cluster'] = labels[i]
    
    # Retrieve cluster centers in the original feature space.
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    # (Each center is a 4D vector: [median_x, median_y, slope, index])
    
    # --- Visualization: 2D Projection (x vs. y) ---
    title='2D Projection (x, y) of N-D Data Clustering using K-Means'
    cf.plot_2d_clustering(data, windows, cf.reassign_clusters, title, centers=centers)
    
# ---------------------------
# Main function using the Elbow method and standard k-medoids from scikit-learn-extra (PAM algorithm)
# ---------------------------
def main_kmedoids_auto():
    # Generate synthetic N-D data (3D: x, y, z)
    np.random.seed(42)
    n_points = 300
    
    data = np.column_stack((x, y,))
    
    # Create windows from the data
    window_size = w_size  # points per window
    windows = create_windows(data, window_size)
    
    # Assign inverted-V block membership using a user-defined block size p.
    p = 4  
    windows = assign_inverted_v_block(windows, p)
    
    # Compute normalization factors:
    # D_max: maximum Euclidean distance between window medians
    D_max = 0
    for i in range(len(windows)):
        for j in range(i+1, len(windows)):
            dist_ij = np.linalg.norm(windows[i]['data'].flatten() - windows[j]['data'].flatten())
            # dist_ij = np.linalg.norm(windows[i]['median'] - windows[j]['median'])
            if dist_ij > D_max:
                D_max = dist_ij
    # T_max: maximum index difference (last window index)
    T_max = windows[-1]['index'] if windows else 1

    # Weight parameters and threshold
    lambda_e = 1.0
    lambda_p = 1.0
    beta = 0.5
    delta = 0.1
    threshold = 1.
    gamma_block = 1.0  # bonus weight for block membership

    if not slope_info:
        lambda_p=0.0
    
    # Create a closure for the distance function
    def dist_func(w1, w2):
        return custom_distance(w1, w2, D_max, T_max, lambda_e, lambda_p, beta,
                               delta, threshold=threshold, gamma_block=gamma_block)

    # Build the precomputed distance matrix
    n_windows = len(windows)
    distance_matrix = np.zeros((n_windows, n_windows))
    for i in range(n_windows):
        for j in range(i, n_windows):
            d = dist_func(windows[i], windows[j])
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d  # symmetry

    
    # Define the clustering function
    kmedoids = KMedoids(metric='precomputed', method='pam',random_state=42)
    visualizer = KElbowVisualizer(kmedoids, k=(2,10),timings=False)
    visualizer.fit(distance_matrix)        # Fit the data to the visualizer
    visualizer.show()                      # Finalize and render the figure
    
    # Set the number of clusters and use standard KMedoids with algorithm 'pam'
    k = visualizer.elbow_value_
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', method='pam', random_state=42)
    kmedoids.fit(distance_matrix)
    labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_

    # Assign cluster labels for visualization
    for i, w in enumerate(windows):
        w['cluster'] = labels[i]
    
    # --- Visualization: 2D Projection (x vs. y) ---
    # --- Visualization: 2D Projection (x vs. y) ---
    title='2D Projection (x, y) of N-D Data Semi-Automated Clustering\n(using KMedoids (PAM) with Normalized & Weighted Metrics)'
    cf.plot_2d_clustering(data, windows, cf.reassign_clusters, title, medoid_indices=medoid_indices)
    
if __name__ == '__main__':
    main_kmedoids()
    main_kmeans()
    main_kmedoids_auto()
