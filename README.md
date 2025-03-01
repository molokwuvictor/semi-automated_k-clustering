# Flow Regime Identification Using K-Means and K-Medoids Clustering

## Summary
This Python script identifies flow regimes present in a pressure transient diagnostic data. It applies both **K-Means** and **K-Medoids** clustering techniques, as well as semi-automated K-clustering using the Elbow method. It can reliably partition the diagnostic data into the different flow regimes by integrating various subject-specific diagnostic indicators. 
The program implements a sliding window approach to segment the pressure transient diagnostic data. Two standard partitioning techniques are utilized:
- **K-Means:** Applies Euclidean distance metric on segments, where each segment is characterized by a set of normalized mid-point (x, y) coordinates, along with slope and index information.
- **K-Medoids:** Applies a weighted composite pair-wise dissimilarity between segments, where is segment is characterized by Euclidean distance, angular difference, temporal and pattern information.

## Main Features

- **Composite-Metric Integration:** Combines several metrics, such as:
  - **Euclidean Distance:** Measures geometric differences.
  - **Angular Difference:** Computes differences in trends (slope) among segments.
  - **Temporal Penalty:** Disincentivizes cluster switching, especially on boundaries (i.e., prevents repeating clusters).
  - **Inverted-V Identification:** Recognizes a specific transient pattern which occurs in the early time region (ETR). This is an inverted-V, which indicates the strong wellbore storage on the left side of       the pattern and the weak wellbore storage on the right side of the pattern.
- **Semi-Automated Cluster Selection:** Applies the Elbow method, a heuristic technique, to find the most suitable number of clusters.
    
## Methodology

### Data Segmentation

The pressure diagnostic data is divided into segments by using a sliding window. Segment-by-segment analysis is more effective for flow regime identification because flow regimes are a cluster of data points (i.e., a segment). Segmentation also helps to reduce noise, making the analysis more credible.

### Feature Computation

For each segment, a set of features is extracted by computing a number of metrics as illustrated in Figure 1.
![Figure 1: 2D dataset with illustrating segmentation yielding two segments with feature vectors *P<sub>i</sub>* and *P<sub>j</sub>* and slopes *m<sub>i</sub>* and *m<sub>j</sub>*](./images/segmentation_1.png)

#### K-Means Clustering

Each segment is expressed as a set of normalized values consisting of:
  - **Mid-point \(x, y\) Coordinates:** The x and y coordinates, and median of all data points in the segment.
  - **Slope:** Derived using linear regression to determine the trend of the segment.
  - **Index Information:** The position of the segment within the dataset.
**Distance Calculation:** The set of normalized values are vertically stacked and used to compute a distance metric using the Euclidean norm. 

#### K-Medoids Clustering

A set of pairwise standardized dissimilarity measures are calculated between segments. These measures equate to various dimensions of the data:
- **Normalized Euclidean Distance**  
  For two 2D segments *P<sub>i</sub>* and *P<sub>j</sub>* each containing *m* data points:
  - Represent *P<sub>i</sub>* as:  
  ![P_i = \{(x_{i1}, y_{i1}), (x_{i2}, y_{i2}), \ldots, (x_{im}, y_{im})\}](https://latex.codecogs.com/svg.latex?P_i%20%3D%20%5C%7B%28x_%7Bi1%7D%2C%20y_%7Bi1%7D%29%2C%20%28x_%7Bi2%7D%2C%20y_%7Bi2%7D%29%2C%20%5Cldots%2C%20%28x_%7Bim%7D%2C%20y_%7Bim%7D%29%5C%7D)
  - Represent *P<sub>j</sub>* as:  
  ![P_j = \{(x_{j1}, y_{j1}), (x_{j2}, y_{j2}), \ldots, (x_{jm}, y_{jm})\}](https://latex.codecogs.com/svg.latex?P_j%20%3D%20%5C%7B%28x_%7Bj1%7D%2C%20y_%7Bj1%7D%29%2C%20%28x_%7Bj2%7D%2C%20y_%7Bj2%7D%29%2C%20%5Cldots%2C%20%28x_%7Bjm%7D%2C%20y_%7Bjm%7D%29%5C%7D)
     
   The normalized Euclidean distance <sub>![\tilde{d}_{E_{ij}}](https://latex.codecogs.com/svg.latex?\tilde{d}_{E_{ij}})</sub> is computed as:

   ![Normalized Euclidean Distance](https://latex.codecogs.com/svg.latex?\tilde{d}_{E_{ij}}=\frac{1}{D_{E_{\max}}}\sqrt{\sum_{k=1}^{m}\Bigl[(x_{ik}-x_{jk})^2+(y_{ik}-y_{jk})^2\Bigr]})

   where *D<sub>E_max</sub>* is the maximum value in the Euclidean distance matrix.
- **Angular Dissimilarity**  
   For segments with at least two points, compute slopes *m₁* and *m₂* (via linear regression) for segments *P<sub>i</sub>* and *P<sub>j</sub>* respectively. The angular dissimilarity <sub>![\tilde{d}_{\theta_{ij}}](https://latex.codecogs.com/svg.latex?\tilde{d}_{\theta_{ij}})</sub> is:

   ![Angular Dissimilarity](https://latex.codecogs.com/svg.latex?\tilde{d}_{\theta_{ij}}=\frac{1}{90^\circ}\arctan\Bigl(\frac{|m_1-m_2|}{1+m_1m_2}\Bigr))

- **Temporal Penalty**  
   Let δ (delta) be the absolute difference between the indices of segments *P<sub>i</sub>* and *P<sub>j</sub>*:

   For a dataset with *n* segments, define *T<sub>max</sub> = n - 1*. Then, the temporal penalty <sub>![\tilde{d}_{T_{ij}}](https://latex.codecogs.com/svg.latex?\tilde{d}_{T_{ij}})</sub> is:
   If *T<sub>max</sub> > 0*:

  ![Temporal Penalty](https://latex.codecogs.com/svg.latex?\tilde{d}_{T_{ij}}=\frac{\max(0,\delta-1)}{T_{max}})
  
  Otherwise:

  ![Temporal Penalty Alternative](https://latex.codecogs.com/svg.latex?\tilde{d}_{T_{ij}}=\max(0,\delta-1))

- **Inverted-V Identification**  
  When an inverted-V pattern is detected before a predefined cut-off (*early_time_index*), boolean labels <sub>![calligraphic w](https://latex.codecogs.com/svg.latex?\mathcal{w})</sub> and <sub>![calligraphic w](https://latex.codecogs.com/svg.latex?\mathcal{w}')</sub> are set to True for the corresponding segments. The dissimilarity <sub>![d_{\Lambda(i,j)}](https://latex.codecogs.com/svg.latex?d_{\Lambda(i,j)})</sub> is defined as:
  <img alt="Figure 2: Illustration of the inverted-V pattern (dashed line) in a diagnostic plot, which has been split into n-segments and overlain by a sliding block of p-segments](./images/segmentation_2.png) 

  ![Inverted-V Identification](https://latex.codecogs.com/svg.latex?%5Ctilde%7Bd%7D_%7B%5CLambda_%7Bij%7D%7D%3D%5Cbegin%7Bcases%7D-1%2C%26%5Ctext%7Bif%20%7Dw%5Ctext%7B%20and%20%7Dw'%5Ctext%7B%20are%20True%7D%5C%5C0%2C%26%5Ctext%7Botherwise%7D%5Cend%7Bcases%7D)

  This ensures that segments with an inverted-V pattern are strongly grouped together.

- **Overall Normalized Dissimilarity**  
  The total dissimilarity <sub>![\tilde{d}_{ij}](https://latex.codecogs.com/svg.latex?\tilde{d}_{ij})</sub> between segments *P<sub>i</sub>* and *P<sub>j</sub>* is computed as a weighted sum of the above metrics:

  ![Overall Normalized Dissimilarity](https://latex.codecogs.com/svg.latex?\tilde{d}_{ij}=\max\Bigl(\lambda_E\tilde{d}_{E_{ij}}+\lambda_{\theta}\tilde{d}_{\theta_{ij}}+\beta_T\tilde{d}_{T_{ij}}+\gamma_{\Lambda}\tilde{d}_{\Lambda_{ij}},0\Bigr))

  where the default hyperparameters are:
  ![\lambda_E = 1](https://latex.codecogs.com/svg.latex?\lambda_E%20=%201), ![\lambda_\theta = 1](https://latex.codecogs.com/svg.latex?\lambda_\theta%20=%201), ![\beta_T = 0.5](https://latex.codecogs.com/svg.latex?\beta_T%20=%200.5) and ![\gamma_\Lambda = 1](https://latex.codecogs.com/svg.latex?\gamma_\Lambda%20=%201)


The max function ensures that the overall dissimilarity is non-negative.

### Clustering Techniques

- **K-Means Clustering:**  
  - **Representation:** Each segment is represented by a feature vector containing normalized mid-point coordinates, slope, and index information.
  - **Process:** The Euclidean distance between the feature vector and the cluster centroid is computed, segments are assigned to the nearest centroid, and centroids are updated iteratively until convergence.

- **K-Medoids Clustering:**  
  - **Representation:** Uses the composite dissimilarity measure <sub>![\tilde{d}_{ij}](https://latex.codecogs.com/svg.latex?\tilde{d}_{ij})</sub> computed from Euclidean, angular, temporal, and inverted-V metrics.
  - **Process:** Selects actual data segments (medoids) as cluster centers and forms clusters based on the weighted aggregation of these dissimilarities.

- **Semi-Automated K-Clustering with the Elbow Method:**  
  - **Process:** Computes the within-cluster sum-of-squares (WCSS) for various \(K\) values, generates an Elbow plot to indicate where WCSS reduction plateaus, and recommends an optimal \(K\) while allowing manual adjustments.

## Installation and Requirements

### Requirements
- **Python 3.x**
- Required packages:
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `pandas`
  - `matplotlib` (for visualization)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/molokwuvictor/Semi-Automated_K-Clustering.git
cd your-repo
pip install -r requirements.txt
