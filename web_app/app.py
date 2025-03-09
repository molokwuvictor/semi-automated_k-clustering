from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
import auxiliary_functions as cf
import os
import traceback

app = Flask(__name__)

# Global variables to store the current data
current_data = None
current_sheet_name = None

def process_data(file_data, sheet_name='Sheet1'):
    """Process the uploaded Excel/CSV file and return the data"""
    try:
        if file_data.filename.endswith('.csv'):
            df = pd.read_csv(file_data)
        else:
            df = pd.read_excel(file_data, sheet_name=sheet_name)
        
        X = df[['lndt', 'dp_dlndt', 'dp']]
        lndt, dp_dlndt, dp = X.iloc[:,0], X.iloc[:,1], X.iloc[:,2]
        
        lndp_dlndt = np.log(dp_dlndt)
        lndp = np.log(dp)
        
        grad = cf.bourdet_derivative(x=lndt, y=dp, L=0.1, transform_x=False, transform_y=False)
        grad_2 = cf.bourdet_derivative(x=lndt, y=lndp_dlndt, L=0., transform_x=False, transform_y=False)
        
        # Normalize the dataset
        grad_2 = cf.min_max_scaler(grad_2, limits=[-1,1])
        lndt = cf.min_max_scaler(lndt, limits=[-1,1])
        lndp_dlndt = cf.min_max_scaler(lndp_dlndt, limits=[-1,1])
        
        return np.array(lndt), np.array(lndp_dlndt), np.array(grad_2)
    except Exception as e:
        return None, None, None

def create_windows(data, window_size):
    """Create windows from the data with regression for slope and median for center"""
    windows = []
    n = data.shape[0]
    for i in range(0, n - window_size + 1, window_size):
        window_data = data[i:i+window_size]
        if window_data.shape[0] < 2:
            break
        median = np.median(window_data, axis=0)
        if np.ptp(window_data[:, 0]) != 0:
            slope, _ = np.polyfit(window_data[:, 0], window_data[:, 1], 1)
        else:
            slope = 0.0
        windows.append({
            'data': window_data,
            'median': median,
            'slope': slope,
            'index': len(windows)
        })
    return windows

def assign_inverted_v_block(windows, p, early_time_index=2):
    """Assign inverted-V block membership for a block of p consecutive windows"""
    for w in windows:
        w['inverted_block'] = False
    
    n = len(windows)
    for i in range(n - p + 1):
        block = windows[i:i+p]
        found = False
        for j in range(p - 1):
            if block[j]['slope'] > 0 and block[j+1]['slope'] < 0:
                found = True
                break
        if found:
            k = 0 if i <= early_time_index else i
            for j in range(k, i+p):
                windows[j]['inverted_block'] = True
    return windows

def custom_distance(w1, w2, D_max, T_max, lambda_e=1.0, lambda_p=1.0, beta=0.5,
                   delta=1.0, threshold=1e-3, gamma_block=1.0):
    """Compute custom distance between two windows"""
    index_diff = abs(w1['index'] - w2['index'])
    
    median_dist = np.linalg.norm(w1['data'].flatten() - w2['data'].flatten())
    normalized_euclid = median_dist / D_max if D_max > 0 else median_dist
    
    if index_diff == 1 and abs(w1['slope']) < threshold and abs(w2['slope']) < threshold:
        return lambda_e * normalized_euclid * 0.1
    
    angle_diff = abs(np.degrees(np.arctan(w1['slope'])) - np.degrees(np.arctan(w2['slope'])))
    if angle_diff > 90:
        angle_diff = 180 - angle_diff
    normalized_angle = angle_diff / 90.0
    
    norm_temporal = (max(0, index_diff - 1) / T_max) if T_max > 0 else max(0, index_diff - 1)
    
    concave_bonus = 0.0
    if index_diff == 1:
        dx = abs(w1['median'][0] - w2['median'][0])
        if dx > 0:
            m1 = w1['slope']
            m2 = w2['slope']
            m_avg = (m1 + m2) / 2.0
            y_dd = (m2 - m1) / dx
            if abs(y_dd) < 1e-6:
                R = float('inf')
            else:
                R = (1 + m_avg**2)**(1.5) / abs(y_dd)
            if R > 2 * dx:
                concave_bonus = -delta
    
    block_bonus = 0.0
    if w1.get('inverted_block', False) and w2.get('inverted_block', False):
        block_bonus = -gamma_block
    
    total_distance = (lambda_e * normalized_euclid + 
                     lambda_p * normalized_angle + 
                     beta * norm_temporal + 
                     concave_bonus + 
                     block_bonus)
    return max(total_distance, 0)

def perform_clustering(method, params):
    """Perform clustering based on the selected method and parameters"""
    global current_data
    if current_data is None:
        return None, None
    
    try:
        x, y, z = current_data
        data = np.column_stack((x, y))
        windows = create_windows(data, params['window_size'])
        # print(f"Created {len(windows)} windows")
        
        if method in ['kmedoids', 'semi_automated']:
            windows = assign_inverted_v_block(windows, params['p'])
            
            D_max = 0
            for i in range(len(windows)):
                for j in range(i+1, len(windows)):
                    dist_ij = np.linalg.norm(windows[i]['data'].flatten() - windows[j]['data'].flatten())
                    if dist_ij > D_max:
                        D_max = dist_ij
            T_max = windows[-1]['index'] if windows else 1
            
            def dist_func(w1, w2):
                return custom_distance(w1, w2, D_max, T_max, 
                                     params['lambda_e'], params['lambda_p'],
                                     params['beta'], params['delta'],
                                     params['threshold'], params['gamma_block'])
            
            n_windows = len(windows)
            distance_matrix = np.zeros((n_windows, n_windows))
            for i in range(n_windows):
                for j in range(i, n_windows):
                    d = dist_func(windows[i], windows[j])
                    distance_matrix[i, j] = d
                    distance_matrix[j, i] = d
            
            if method == 'semi_automated':
                try:
                    # Create and fit the visualizer
                    kmedoids = KMedoids(metric='precomputed', method='pam', random_state=42)
                    visualizer = KElbowVisualizer(kmedoids, k=(2,params['n_clusters']*3), timings=False)
                    visualizer.fit(distance_matrix)
                    
                    # Instead of creating a base64 image, extract the data for JS plotting
                    elbow_data = {
                        'k_values': visualizer.k_values_, 
                        'k_scores': visualizer.k_scores_,
                        'elbow_value': int(visualizer.elbow_value_) if visualizer.elbow_value_ is not None else None,
                        'elbow_score': float(visualizer.elbow_score_) if visualizer.elbow_score_ is not None else None,
                        'locate_elbow': bool(visualizer.locate_elbow),
                        'estimator': str(visualizer.estimator),
                    }
                    
                    # print(f"Elbow data: {elbow_data}")
                    
                    k = visualizer.elbow_value_
                    if k is None:  # If no clear elbow is found
                        k = params['n_clusters']
                    # print(f"Optimal k from elbow method: {k}")
                except Exception as e:
                    print(f"Error in elbow plot generation: {str(e)}")
                    traceback.print_exc()  # Print the full traceback
                    elbow_data = None
                    k = params['n_clusters']  # Fallback to user-specified number of clusters
            else:
                elbow_data = None
                k = params['n_clusters']
                
            # print(f"Using k={k} for clustering")
            
            # Perform k-medoids clustering
            kmedoids = KMedoids(n_clusters=k, metric='precomputed', method='pam', random_state=42)
            kmedoids.fit(distance_matrix)
            labels = kmedoids.labels_
            medoid_indices = kmedoids.medoid_indices_
            
            # print(f"Original cluster labels: {labels}")
            
            # Assign cluster labels to windows
            for i, label in enumerate(labels):
                windows[i]['cluster'] = int(label)
            
            # Reassign clusters based on x-coordinate ordering
            old_to_new = {}
            clusters = np.unique(labels)
            
            # Compute a representative x value for each cluster
            cluster_repr = {}
            for cluster in clusters:
                xs = np.concatenate([windows[i]['data'][:, 0] for i in range(len(windows)) 
                                    if windows[i]['cluster'] == cluster])
                cluster_repr[cluster] = np.median(xs)
            
            # Sort clusters by their median x value (leftmost first)
            sorted_clusters = sorted(cluster_repr, key=cluster_repr.get)
            
            # Create a mapping from original to new labels
            for new_idx, old_idx in enumerate(sorted_clusters):
                old_to_new[old_idx] = new_idx
                
            # print(f"Cluster mapping: {old_to_new}")
            
            # Apply new labels
            for window in windows:
                window['cluster'] = old_to_new[window['cluster']]
            
            # Extract updated labels
            labels = [window['cluster'] for window in windows]
            
            # print(f"Reassigned cluster labels: {labels}")
            # print(f"Unique reassigned clusters: {np.unique(labels)}")
            
        else:  # kmeans
            # Get parameters with default values
            lambda_e = params.get('lambda_e', 1.0)
            lambda_p = params.get('lambda_p', 1.0)
            beta = params.get('beta', 0.5)
            
            # Convert to float to ensure we handle numeric values correctly
            try:
                lambda_e = float(lambda_e)
                lambda_p = float(lambda_p)
                beta = float(beta)
            except (ValueError, TypeError):
                print("Error converting parameters to float, using defaults")
                lambda_e = 1.0
                lambda_p = 1.0
                beta = 0.5
            
            # print(f"K-means parameters: lambda_e={lambda_e}, lambda_p={lambda_p}, beta={beta}")
            
            # Extract features from windows
            features = []
            for w in windows:
                median = w['median']
                # Apply weighting factors to features
                features.append([
                    lambda_e * median[0],  # Weight x-coordinate
                    lambda_e * median[1],  # Weight y-coordinate
                    lambda_p * w['slope'],  # Weight slope
                    beta * w['index']  # Weight index
                ])
            
            # Print a sample of weighted features for debugging
            # if features:
            #     print(f"Sample weighted feature: {features[0]}")
            
            # Convert to numpy array
            X = np.array(features)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=params['n_clusters'], random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            
            # print(f"Original KMeans cluster labels: {labels}")
            
            # Assign cluster labels to windows
            for i, label in enumerate(labels):
                windows[i]['cluster'] = int(label)
            
            # Reassign clusters based on x-coordinate ordering
            old_to_new = {}
            clusters = np.unique(labels)
            
            # Compute a representative x value for each cluster
            cluster_repr = {}
            for cluster in clusters:
                xs = np.concatenate([windows[i]['data'][:, 0] for i in range(len(windows)) 
                                    if windows[i]['cluster'] == cluster])
                cluster_repr[cluster] = np.median(xs)
            
            # Sort clusters by their median x value (leftmost first)
            sorted_clusters = sorted(cluster_repr, key=cluster_repr.get)
            
            # Create a mapping from original to new labels
            for new_idx, old_idx in enumerate(sorted_clusters):
                old_to_new[old_idx] = new_idx
                
            # print(f"KMeans cluster mapping: {old_to_new}")
            
            # Apply new labels
            for window in windows:
                window['cluster'] = old_to_new[window['cluster']]
            
            # Extract updated labels
            labels = [window['cluster'] for window in windows]
            
            # print(f"Reassigned KMeans cluster labels: {labels}")
            # print(f"Unique reassigned KMeans clusters: {np.unique(labels)}")
            
            medoid_indices = None
            elbow_data = None
        
        # Prepare data for plotting
        plot_data = {
            'x': x.tolist(),
            'y': y.tolist(),
            'windows': [{
                'data': w['data'].tolist(),  # This is the complete segment data
                'median': w['median'].tolist(),
                'cluster': int(w['cluster']),  # Cluster label for the entire segment
                'index': w['index']  # Add index to maintain ordering
            } for w in windows],
            'labels': [int(w['cluster']) for w in windows],  # Use window cluster labels
            'medoid_indices': medoid_indices.tolist() if medoid_indices is not None else None,
            'centers': centers.tolist() if 'centers' in locals() else None
        }
        
        # Add debug information
        # print("Plot data preparation:")
        # print(f"Number of windows: {len(plot_data['windows'])}")
        # print(f"Unique cluster labels in windows: {np.unique([w['cluster'] for w in plot_data['windows']])}")
        # print(f"Window data shapes: {[len(w['data']) for w in plot_data['windows'][:5]]} (first 5 windows)")
        
        return plot_data, elbow_data
    except Exception as e:
        # print(f"Error in perform_clustering: {str(e)}")
        traceback.print_exc()  # Print the full traceback
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data, current_sheet_name
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    sheet_name = request.form.get('sheet_name', 'Sheet1')
    current_sheet_name = sheet_name
    
    x, y, z = process_data(file, sheet_name)
    if x is None:
        return jsonify({'error': 'Error processing file'}), 400
    
    current_data = (x, y, z)
    return jsonify({'message': 'File uploaded successfully'})

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        data = request.json
        method = data.get('method', 'kmeans')
        params = data
        
        # Debug log the parameters
        # print(f"Clustering method: {method}")
        # print(f"Parameters: {params}")
        # print(f"Lambda_E: {params.get('lambda_e', 1.0)}")
        # print(f"Lambda_P: {params.get('lambda_p', 1.0)}")
        # print(f"Beta: {params.get('beta', 0.5)}")
        
        global current_data, current_sheet_name
        
        if current_data is None:
            return jsonify({'error': 'No data uploaded yet'}), 400
        
        plot_data, elbow_data = perform_clustering(method, params)
        if plot_data is None:
            return jsonify({'error': 'Error performing clustering'}), 400
        
        # print(f"Clustering completed. Elbow plot generated: {elbow_data is not None}")
        # if elbow_data is not None:
        #     print(f"Elbow plot data length: {len(elbow_data)}")
        
        return jsonify({
            'plot_data': plot_data,
            'elbow_data': elbow_data
        })
    except Exception as e:
        print(f"Error in cluster route: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 