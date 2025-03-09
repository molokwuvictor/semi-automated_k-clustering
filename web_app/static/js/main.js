document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const sheetNameInput = document.getElementById('sheetName');
    const clusteringMethod = document.getElementById('clusteringMethod');
    const kmedoidsParams = document.querySelector('.kmedoids-params');
    const extraMetricsToggle = document.getElementById('extraMetricsToggle');
    const extraMetricsParams = document.querySelector('.extra-metrics-params');
    const updatePlotBtn = document.getElementById('updatePlot');
    const elbowPlotContainer = document.getElementById('elbowPlotContainer');

    // Slider elements
    const sliders = {
        nClusters: document.getElementById('nClusters'),
        windowSize: document.getElementById('windowSize'),
        lambdaE: document.getElementById('lambdaE'),
        lambdaP: document.getElementById('lambdaP'),
        beta: document.getElementById('beta'),
        gammaBlock: document.getElementById('gammaBlock'),
        p: document.getElementById('p'),
        // Extra metrics sliders
        delta: document.getElementById('delta'),
        threshold: document.getElementById('threshold')
    };

    // Plot elements
    const clusterPlot = document.getElementById('clusterPlot');
    const elbowPlot = document.getElementById('elbowPlot');

    // Initialize plots
    let clusterPlotInstance = null;
    let elbowPlotInstance = null;

    // Event Listeners
    uploadForm.addEventListener('submit', handleFileUpload);
    clusteringMethod.addEventListener('change', handleMethodChange);
    updatePlotBtn.addEventListener('click', updatePlots);
    extraMetricsToggle.addEventListener('change', toggleExtraMetrics);
    
    // Add event listeners to all sliders
    Object.entries(sliders).forEach(([key, slider]) => {
        if (slider) {  // Check if slider exists
            slider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                e.target.nextElementSibling.textContent = value.toFixed(1);
            });
        }
    });

    // Function to toggle extra metrics visibility
    function toggleExtraMetrics(e) {
        console.log("Extra metrics toggle changed:", e.target.checked);
        extraMetricsParams.style.display = e.target.checked ? 'block' : 'none';
    }

    // Initialize UI state
    document.addEventListener('DOMContentLoaded', () => {
        // Set initial state based on selected method
        handleMethodChange({ target: clusteringMethod });
    });

    // Method Change Handler
    function handleMethodChange(e) {
        const method = e.target.value;
        console.log("Method changed to:", method);
        
        // Show/hide appropriate parameter sections
        const showKmedoidsParams = method !== 'kmeans';
        
        // For K-means, we still show Lambda_E, Lambda_P, and Beta, but hide the rest
        const lambdaESlider = document.querySelector('.slider-group:has(#lambdaE)');
        const lambdaPSlider = document.querySelector('.slider-group:has(#lambdaP)');
        const betaSlider = document.querySelector('.slider-group:has(#beta)');
        const gammaBlockSlider = document.querySelector('.slider-group:has(#gammaBlock)');
        const pSlider = document.querySelector('.slider-group:has(#p)');
        const extraMetricsToggleParent = extraMetricsToggle.closest('.extra-metrics-toggle');
        
        // Always show Lambda_E, Lambda_P, and Beta
        if (lambdaESlider) lambdaESlider.style.display = 'block';
        if (lambdaPSlider) lambdaPSlider.style.display = 'block';
        if (betaSlider) betaSlider.style.display = 'block';
        
        // Only show these for k-medoids and semi-automated
        if (gammaBlockSlider) gammaBlockSlider.style.display = showKmedoidsParams ? 'block' : 'none';
        if (pSlider) pSlider.style.display = showKmedoidsParams ? 'block' : 'none';
        extraMetricsToggleParent.style.display = showKmedoidsParams ? 'block' : 'none';
        
        // Move Lambda_E, Lambda_P, and Beta out of kmedoids-params if they're not already
        const parameterSliders = document.querySelector('.parameter-sliders');
        const kmedoidsParams = document.querySelector('.kmedoids-params');
        
        // Hide the rest of kmedoids params for kmeans
        kmedoidsParams.style.display = showKmedoidsParams ? 'block' : 'none';
        
        if (!showKmedoidsParams) {
            extraMetricsToggle.checked = false;
            extraMetricsParams.style.display = 'none';
        } else {
            // If kmedoids or semi-automated, show/hide extra metrics based on checkbox
            extraMetricsParams.style.display = extraMetricsToggle.checked ? 'block' : 'none';
        }
        
        elbowPlotContainer.style.display = 'none';
    }

    // File Upload Handler
    async function handleFileUpload(e) {
        e.preventDefault();
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('sheet_name', sheetNameInput.value);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                updatePlotBtn.disabled = false;
                showNotification('File uploaded successfully!', 'success');
            } else {
                showNotification(data.error || 'Error uploading file', 'error');
            }
        } catch (error) {
            showNotification('Error uploading file: ' + error.message, 'error');
        }
    }

    // Update Plots Handler
    async function updatePlots() {
        const method = clusteringMethod.value;
        const params = {
            method: method,
            n_clusters: parseInt(sliders.nClusters.value),
            window_size: parseInt(sliders.windowSize.value),
            // Always include these parameters for all methods
            lambda_e: parseFloat(sliders.lambdaE.value),
            lambda_p: parseFloat(sliders.lambdaP.value),
            beta: parseFloat(sliders.beta.value)
        };

        // Add additional parameters for k-medoids and semi-automated
        if (method !== 'kmeans') {
            const extraMetricsEnabled = extraMetricsToggle.checked;
            Object.assign(params, {
                delta: extraMetricsEnabled ? parseFloat(sliders.delta.value) : 0.1,
                threshold: extraMetricsEnabled ? parseFloat(sliders.threshold.value) : 0.1,
                gamma_block: parseFloat(sliders.gammaBlock.value),
                p: parseInt(sliders.p.value)
            });
        }

        try {
            const response = await fetch('/cluster', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });

            const data = await response.json();
            
            if (response.ok) {
                updateClusterPlot(data.plot_data);
                
                // Handle elbow plot
                if (method === 'semi_automated') {
                    console.log("Elbow plot data received:", data.elbow_plot ? "Yes" : "No");
                    
                    if (data.elbow_plot) {
                        elbowPlotContainer.style.display = 'block';
                        updateElbowPlot(data.elbow_plot);
                        showNotification('Elbow plot generated. Using optimal number of clusters.', 'success');
                    } else {
                        elbowPlotContainer.style.display = 'none';
                        showNotification('Elbow plot generation failed. Using specified number of clusters.', 'warning');
                    }
                } else {
                    elbowPlotContainer.style.display = 'none';
                }
            } else {
                showNotification(data.error || 'Error updating plots', 'error');
            }
        } catch (error) {
            console.error("Error during plot update:", error);
            showNotification('Error updating plots: ' + error.message, 'error');
        }
    }

    // Plot Update Functions
    function updateClusterPlot(plotData) {
        console.log("Updating cluster plot with data:", plotData);
        
        // Clear the plot container
        const clusterPlotElement = document.getElementById('clusterPlot');
        clusterPlotElement.innerHTML = '';
        
        // Define a color palette
        const colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ];
        
        // Prepare the traces for plotting
        const traces = [];
        
        // Group windows by cluster
        const clusterGroups = {};
        plotData.windows.forEach(window => {
            const clusterLabel = window.cluster;
            if (!clusterGroups[clusterLabel]) {
                clusterGroups[clusterLabel] = [];
            }
            clusterGroups[clusterLabel].push(window);
        });
        
        // Sort clusters by their labels to ensure consistent coloring
        const sortedClusters = Object.keys(clusterGroups).sort((a, b) => parseInt(a) - parseInt(b));
        
        // For each cluster, create traces for its segments
        sortedClusters.forEach((clusterLabel, clusterIndex) => {
            const color = colors[clusterIndex % colors.length];
            const clusterName = `Cluster ${parseInt(clusterLabel) + 1}`;
            
            // Create a dummy point for the legend
            traces.push({
                x: [null],
                y: [null],
                mode: 'lines+markers',
                name: clusterName,
                line: { color: color, width: 2 },
                marker: { color: color },
                legendgroup: clusterName,
                showlegend: true
            });
            
            // Add traces for each window segment in this cluster
            clusterGroups[clusterLabel].forEach(window => {
                traces.push({
                    x: window.data.map(d => d[0]),
                    y: window.data.map(d => d[1]),
                    mode: 'lines+markers',
                    name: clusterName,
                    line: { color: color, width: 2 },
                    marker: {
                        color: color,
                        size: 4,
                        symbol: 'circle'
                    },
                    legendgroup: clusterName,
                    showlegend: false
                });
            });
        });
        
        // Add medoid points if available
        if (plotData.medoid_indices && plotData.medoid_indices.length > 0) {
            const medoidPoints = plotData.medoid_indices.map(i => {
                if (i < plotData.windows.length) {
                    return plotData.windows[i].median;
                }
                return null;
            }).filter(p => p !== null);
            
            if (medoidPoints.length > 0) {
                traces.push({
                    x: medoidPoints.map(m => m[0]),
                    y: medoidPoints.map(m => m[1]),
                    mode: 'markers',
                    name: 'Medoids',
                    marker: { 
                        size: 12, 
                        symbol: 'star',
                        color: '#000000',
                        line: { color: '#ffffff', width: 1 }
                    },
                    type: 'scatter'
                });
            }
        }
        
        // Add cluster centers if available
        if (plotData.centers && plotData.centers.length > 0) {
            traces.push({
                x: plotData.centers.map(c => c[0]),
                y: plotData.centers.map(c => c[1]),
                mode: 'markers',
                name: 'Centers',
                marker: { 
                    size: 12, 
                    symbol: 'star',
                    color: '#000000',
                    line: { color: '#ffffff', width: 1 }
                },
                type: 'scatter'
            });
        }
        
        const layout = {
            title: {
                text: 'Clustering Results',
                font: { size: 20, family: 'Roboto' }
            },
            xaxis: { 
                title: { text: 'ln(Δt)', font: { size: 14 } },
                showgrid: true,
                gridcolor: '#e0e0e0'
            },
            yaxis: { 
                title: { text: 'ln(dΔp/dlnΔt)', font: { size: 14 } },
                showgrid: true,
                gridcolor: '#e0e0e0'
            },
            showlegend: true,
            legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: -0.2,
                xanchor: 'center',
                x: 0.5
            },
            margin: { t: 50, b: 100, l: 80, r: 50 },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff'
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };
        
        console.log("Plotting with traces:", traces);
        Plotly.newPlot(clusterPlotElement, traces, layout, config);
    }

    function updateElbowPlot(base64Image) {
        if (!base64Image) {
            console.error("No base64 image data received for elbow plot");
            return;
        }
        
        console.log("Updating elbow plot with image data of length:", base64Image.length);
        
        // Create an img element to display the elbow plot
        const elbowPlotElement = document.getElementById('elbowPlot');
        elbowPlotElement.innerHTML = ''; // Clear previous content
        
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + base64Image;
        img.style.width = '100%';
        img.style.height = 'auto';
        img.style.maxWidth = '800px';
        img.style.display = 'block';
        img.style.margin = '0 auto';
        
        img.onerror = function() {
            console.error("Error loading elbow plot image");
            showNotification("Failed to display elbow plot", "error");
        };
        
        elbowPlotElement.appendChild(img);
    }

    // Notification function
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Remove notification after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}); 