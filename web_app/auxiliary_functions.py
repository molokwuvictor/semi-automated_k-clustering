# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:08:19 2025

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

def bourdet_derivative(x, y, L=0.1, transform_x=False, transform_y=False, remove_end_points=False):
    """
    Compute the Bourdet derivative for given arrays x and y.
    
    Parameters:
        x (array-like): Array of x values.
        y (array-like): Array of y values.
        L (float): Smoothing factor used to select neighboring points.
        transform_x (bool): If True, apply logarithmic transformation to x.
        transform_y (bool): If True, apply logarithmic transformation to y.
        remove_end_points (bool): If True, adjusts endpoint derivatives (currently commented out).
    
    Returns:
        np.ndarray: Array of derivative values computed for each point.
    """
    # Convert inputs to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Optionally apply logarithmic transformation
    if transform_x:
        x = np.log(x)
    if transform_y:
        y = np.log(y)
    
    def forward_difference(y_right, y_center, x_right, x_center):
        """Compute the forward difference derivative."""
        numerator = y_right - y_center
        denominator = x_right - x_center
        return np.divide(numerator, denominator, 
                         out=np.zeros_like(numerator, dtype=float), 
                         where=denominator != 0)

    def backward_difference(y_center, y_left, x_center, x_left):
        """Compute the backward difference derivative."""
        numerator = y_center - y_left
        denominator = x_center - x_left
        return np.divide(numerator, denominator, 
                         out=np.zeros_like(numerator, dtype=float), 
                         where=denominator != 0)
    
    def higher_order_end_point(y_left, y_center, y_right, x_left, x_center, x_right, point=0):
        """
        Compute a second-order difference for an endpoint.
        
        Parameters:
            point (int): 0 for the left endpoint (forward difference),
                         1 for the right endpoint (backward difference).
        """
        if point == 0:
            numerator = -3 * y_left + 4 * y_center - y_right
        else:
            numerator = 3 * y_right - 4 * y_center + y_left
        denominator = x_right - x_left
        return np.divide(numerator, denominator, 
                         out=np.zeros_like(numerator, dtype=float), 
                         where=denominator != 0)
    
    derivatives = []
    eps = 1e-4  # A small constant to avoid numerical issues
    
    for i in range(len(x)):
        # Calculate the difference between current x and all other x values
        diff = x - x[i]
        x_current = x[i]
        y_current = y[i]
        
        if 0 < i < len(x) - 1:
            # Compute thresholds for left and right neighbors
            min_diff = diff.min()
            max_diff = diff.max()
            left_threshold = max(-L, min_diff + eps)
            right_threshold = min(L, max_diff - eps)
            
            # Find the best left neighbor: the largest x value less than x_current
            left_indices = np.where(diff < left_threshold)[0]
            if left_indices.size > 0:
                L_idx = left_indices[np.argmax(x[left_indices])]
            else:
                L_idx = i
            
            # Find the best right neighbor: the smallest x value greater than x_current
            right_indices = np.where(diff > right_threshold)[0]
            if right_indices.size > 0:
                R_idx = right_indices[np.argmin(x[right_indices])]
            else:
                R_idx = i
            
            x_left, y_left = x[L_idx], y[L_idx]
            x_right, y_right = x[R_idx], y[R_idx]
            
            # Compute forward and backward differences
            fd = forward_difference(y_right, y_current, x_right, x_current)
            bd = backward_difference(y_current, y_left, x_current, x_left)
            # Combine the differences using weighted average
            derivative = ((x_right - x_current) * bd + (x_current - x_left) * fd) / (x_right - x_left)
        
        elif i == 0:
            # For the first point, use the forward difference
            max_diff = diff.max()
            right_threshold = min(L, max_diff - eps)
            right_indices = np.where(diff > right_threshold)[0]
            if right_indices.size > 0:
                R_idx = right_indices[np.argmin(x[right_indices])]
            else:
                R_idx = i
            x_right, y_right = x[R_idx], y[R_idx]
            derivative = forward_difference(y_right, y_current, x_right, x_current)
        
        else:  # i == len(x) - 1
            # For the last point, use the backward difference
            min_diff = diff.min()
            left_threshold = max(-L, min_diff + eps)
            left_indices = np.where(diff < left_threshold)[0]
            if left_indices.size > 0:
                L_idx = left_indices[np.argmax(x[left_indices])]
            else:
                L_idx = i
            x_left, y_left = x[L_idx], y[L_idx]
            derivative = backward_difference(y_current, y_left, x_current, x_left)
        
        derivatives.append(derivative)
    
    return np.array(derivatives)

def min_max_scaler(x, limits=[-1,1]):
    """Scale data to a specified range using min-max scaling."""
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    return ((x-x_min)/(x_max-x_min))*(limits[1]-limits[0])+limits[0]

def create_subsequences(time_series, window_size, step_size):
    """Create subsequences from a time series."""
    return np.array([time_series[i:i + window_size] for i in range(0, len(time_series) - window_size + 1, step_size)], dtype=np.double)

def reassign_clusters(windows):
    """
    Reassign clusters so that labels increase from left to right
    based on the median x coordinate of each cluster.
    """
    # Find unique cluster labels.
    clusters = np.unique([w['cluster'] for w in windows])
    
    # Compute a representative x value (here, the median) for each cluster.
    cluster_repr = {}
    for cluster in clusters:
        # Collect all x coordinates for the current cluster.
        xs = np.concatenate([w['data'][:, 0] for w in windows if w['cluster'] == cluster])
        cluster_repr[cluster] = np.median(xs)
    
    # Sort clusters by their median x value (leftmost first).
    sorted_clusters = sorted(cluster_repr, key=cluster_repr.get)
    
    # Create a mapping from the original label to a new, ordered label.
    new_label_mapping = {old: new for new, old in enumerate(sorted_clusters)}
    
    # Reassign the cluster labels.
    for w in windows:
        w['cluster'] = new_label_mapping[w['cluster']]
    
    return windows 