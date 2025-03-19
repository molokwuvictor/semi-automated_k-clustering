document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const searchSidebar = document.querySelector('.search-sidebar');
    const searchToggle = document.querySelector('.search-toggle');
    const closeSidebar = document.querySelector('.close-sidebar');
    const searchTabs = document.querySelectorAll('.search-tab');
    const searchInput = document.querySelector('.search-input');
    const searchButton = document.getElementById('searchButton');
    const searchResults = document.querySelector('.search-results');
    const container = document.querySelector('.container');
    const stopButton = document.querySelector('.stop-button');
    
    // Model dropdown elements
    const modelDropdownContainer = document.getElementById('modelDropdownContainer');
    const modelDropdownButton = document.getElementById('modelDropdownButton');
    const modelDropdownMenu = document.getElementById('modelDropdownMenu');
    const selectedModelText = document.getElementById('selectedModel');
    const modelOptions = document.querySelectorAll('.model-option');

    // State
    let currentTab = 'web';
    let selectedModel = 'deepseek/deepseek-r1:free';
    let isSearching = false;
    let currentController = null; // For aborting fetch requests

    // Event Listeners
    searchToggle.addEventListener('click', toggleSidebar);
    closeSidebar.addEventListener('click', toggleSidebar);
    searchTabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
    
    // Search button click event
    searchButton.addEventListener('click', performSearch);
    
    // Stop button click event
    stopButton.addEventListener('click', stopSearch);
    
    // Enter key in search input
    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !isSearching) {
            performSearch();
        }
    });
    
    // Model dropdown events
    modelDropdownButton.addEventListener('click', toggleModelDropdown);
    
    // Model option selection
    modelOptions.forEach(option => {
        option.addEventListener('click', function() {
            selectModel(this.dataset.model, this.textContent);
            toggleModelDropdown();
        });
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (!modelDropdownButton.contains(e.target) && !modelDropdownMenu.contains(e.target)) {
            modelDropdownMenu.classList.remove('show');
        }
    });

    // Functions
    function toggleSidebar() {
        searchSidebar.classList.toggle('active');
        container.classList.toggle('sidebar-active');
        if (searchSidebar.classList.contains('active')) {
            searchInput.focus();
        }
        
        // Trigger resize event for plots
        const clusterPlot = document.getElementById('clusterPlot');
        const elbowPlot = document.getElementById('elbowPlot');
        
        // Use setTimeout to ensure the DOM has updated before resizing
        setTimeout(() => {
            if (clusterPlot) {
                Plotly.Plots.resize(clusterPlot);
            }
            if (elbowPlot) {
                Plotly.Plots.resize(elbowPlot);
            }
        }, 300); // Wait for transition to complete (300ms matches the CSS transition time)
    }

    function switchTab(tab) {
        currentTab = tab;
        searchTabs.forEach(t => t.classList.remove('active'));
        document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
        searchResults.innerHTML = ''; // Clear results when switching tabs
        
        // Show/hide model dropdown based on tab
        modelDropdownContainer.style.display = tab === 'ai' ? 'block' : 'none';
        
        // Update placeholder based on tab
        if (tab === 'web') {
            searchInput.placeholder = "Search the web...";
            document.querySelector('.web-footnote').classList.add('active');
            document.querySelector('.ai-footnote').classList.remove('active');
        } else {
            searchInput.placeholder = "Ask AI a question...";
            document.querySelector('.web-footnote').classList.remove('active');
            document.querySelector('.ai-footnote').classList.add('active');
        }
    }
    
    function toggleModelDropdown() {
        modelDropdownMenu.classList.toggle('show');
    }
    
    function selectModel(modelId, modelName) {
        selectedModel = modelId;
        selectedModelText.textContent = modelName;
        
        // Update selected class
        modelOptions.forEach(option => {
            if (option.dataset.model === modelId) {
                option.classList.add('selected');
            } else {
                option.classList.remove('selected');
            }
        });
    }

    function stopSearch() {
        if (currentController) {
            currentController.abort();
            currentController = null;
        }
        isSearching = false;
        stopButton.style.display = 'none';
        searchButton.innerHTML = '<i class="fas fa-search"></i>';
        searchButton.disabled = false;
    }

    async function performSearch() {
        const query = searchInput.value.trim();
        
        if (query.length === 0 || isSearching) return;
        
        isSearching = true;
        stopButton.style.display = 'block';
        
        // Update button to show loading state
        const originalButtonContent = searchButton.innerHTML;
        searchButton.innerHTML = '<div class="loading-spinner"></div>';
        searchButton.disabled = true;
        
        try {
            // Create new AbortController for this request
            currentController = new AbortController();
            const signal = currentController.signal;
            
            if (currentTab === 'web') {
                searchResults.innerHTML = '<div class="search-result-item"><div class="search-result-content">Searching...</div></div>';
                
                const response = await fetch(`/web_search?query=${encodeURIComponent(query)}`, { signal });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Failed to perform web search');
                }
                
                const data = await response.json();
                displayWebResults(data);
            } else {
                // Initialize the AI result container with loading state and sections
                initializeAIResultContainer();
                
                const response = await fetch('/ai_search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ 
                        query: query,
                        model: selectedModel,
                        stream: true // Request streaming response
                    }),
                    signal
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Failed to perform AI search');
                }
                
                // Setup for streaming response
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';
                
                // Process the stream
                while (true) {
                    const { done, value } = await reader.read();
                    
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    
                    // Process complete JSON objects in buffer
                    let startIndex = 0;
                    let endIndex;
                    
                    while ((endIndex = buffer.indexOf('\n', startIndex)) !== -1) {
                        const jsonChunk = buffer.substring(startIndex, endIndex).trim();
                        startIndex = endIndex + 1;
                        
                        if (jsonChunk) {
                            try {
                                const data = JSON.parse(jsonChunk);
                                processStreamedData(data);
                            } catch (e) {
                                console.error('Error parsing JSON chunk:', e, jsonChunk);
                            }
                        }
                    }
                    
                    // Keep the remainder in the buffer
                    buffer = buffer.substring(startIndex);
                }
                
                // Process any remaining data in buffer
                if (buffer.trim()) {
                    try {
                        const data = JSON.parse(buffer.trim());
                        processStreamedData(data);
                    } catch (e) {
                        console.error('Error parsing final JSON chunk:', e, buffer);
                    }
                }
                
                // Finalize the display
                finalizeAIResult();
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                searchResults.innerHTML = `
                    <div class="search-result-item">
                        <div class="search-result-content">Search cancelled</div>
                    </div>
                `;
            } else {
                console.error("Search error:", error);
                searchResults.innerHTML = `
                    <div class="search-result-item error">
                        <div class="search-result-content">Error performing search: ${error.message}</div>
                    </div>
                `;
            }
        } finally {
            // Restore button state
            searchButton.innerHTML = originalButtonContent;
            searchButton.disabled = false;
            isSearching = false;
            stopButton.style.display = 'none';
            currentController = null;
        }
    }

    // Initialize AI result container with empty sections
    function initializeAIResultContainer() {
        const resultHTML = `
            <div class="search-result-item ai-result">
                <div class="model-info">
                    <i class="fas fa-robot"></i> ${selectedModelText.textContent}
                </div>
                <div class="thinking-section">
                    <div class="thinking-header">
                        <i class="fas fa-brain"></i> Thinking Process <span class="wavy-loader"></span>
                    </div>
                    <div class="thinking-content">
                    </div>
                </div>
                <div class="answer-section" style="display: none;">
                    <div class="answer-header">
                        <i class="fas fa-comment-dots"></i> Answer
                    </div>
                    <div class="answer-content"></div>
                </div>
            </div>
        `;
        searchResults.innerHTML = resultHTML;
    }

    // Process streamed data tokens
    function processStreamedData(data) {
        const thinkingContent = document.querySelector('.thinking-content');
        const answerSection = document.querySelector('.answer-section');
        const answerContent = document.querySelector('.answer-content');
        const wavyLoader = document.querySelector('.wavy-loader');
        
        // Remove wavy loader when the first thinking token arrives
        if (data.token_type === 'thinking' && data.token && wavyLoader) {
            wavyLoader.style.display = 'none';
        }
        
        if (data.token_type === 'thinking' && data.token) {
            // For thinking content, use a more efficient approach
            // Append thinking token to a temporary variable instead of directly to the DOM
            const formattedToken = formatToken(data.token);
            
            // Create a document fragment for better performance
            const fragment = document.createDocumentFragment();
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = formattedToken;
            
            // Move all nodes from the temp div to the fragment
            while (tempDiv.firstChild) {
                fragment.appendChild(tempDiv.firstChild);
            }
            
            // Append the fragment to the DOM (single operation)
            thinkingContent.appendChild(fragment);
            
            // Auto-scroll to bottom of thinking section
            thinkingContent.scrollTop = thinkingContent.scrollHeight;
            
            // Render LaTeX if needed, but less frequently for better performance
            if (data.token.includes('$') || data.token.includes('\n')) {
                renderMathJax();
            }
        } 
        else if (data.token_type === 'answer') {
            // Show answer section if hidden
            if (answerSection.style.display === 'none') {
                answerSection.style.display = 'block';
            }
            
            // Append answer token
            if (data.token) {
                const formattedToken = formatToken(data.token);
                
                // Same efficient approach for answer content
                const fragment = document.createDocumentFragment();
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = formattedToken;
                
                while (tempDiv.firstChild) {
                    fragment.appendChild(tempDiv.firstChild);
                }
                
                answerContent.appendChild(fragment);
                
                // Auto-scroll to bottom of answer section
                answerContent.scrollTop = answerContent.scrollHeight;
                
                // Render LaTeX if needed, but less frequently
                if (data.token.includes('$') || data.token.includes('\n')) {
                    renderMathJax();
                }
            }
        }
        
        // Handle model info if provided
        if (data.model_used) {
            const modelInfo = document.querySelector('.model-info');
            if (modelInfo) {
                modelInfo.innerHTML = `<i class="fas fa-robot"></i> ${data.model_used}`;
            }
        }
    }

    // Format individual tokens for display
    function formatToken(token) {
        // Special handling for newlines
        if (token === '\n') {
            return '<br>';
        }
        
        // If token contains $ signs, it might be part of LaTeX equation
        // We need to handle this carefully during streaming
        if (token.includes('$')) {
            // Get the current content of the container (using a more efficient approach)
            const thinkingContent = document.querySelector('.thinking-content');
            const answerContent = document.querySelector('.answer-content');
            
            const thinkingHTML = thinkingContent.innerHTML;
            const answerHTML = answerContent ? answerContent.innerHTML : '';
            
            // Check if we're in the middle of a LaTeX equation
            const inThinkingLatexEquation = thinkingHTML.includes('<div class="latex-equation">$$') && 
                                           !thinkingHTML.endsWith('$$</div>');
            const inThinkingLatexInline = thinkingHTML.includes('<span class="latex-inline">$') && 
                                         !thinkingHTML.endsWith('$</span>');
            const inAnswerLatexEquation = answerHTML.includes('<div class="latex-equation">$$') && 
                                         !answerHTML.endsWith('$$</div>');
            const inAnswerLatexInline = answerHTML.includes('<span class="latex-inline">$') && 
                                       !answerHTML.endsWith('$</span>');
            
            // Handle completion of LaTeX delimiters
            if ((inThinkingLatexEquation || inAnswerLatexEquation) && token.includes('$$')) {
                return token.replace('$$', '$$</div>');
            } else if ((inThinkingLatexInline || inAnswerLatexInline) && token.includes('$')) {
                return token.replace('$', '$</span>');
            } else if (token === '$$') {
                return '<div class="latex-equation">$$';
            } else if (token === '$') {
                return '<span class="latex-inline">$';
            }
        }
        
        // Handle code blocks (start/end)
        if (token === '```') {
            const thinkingContent = document.querySelector('.thinking-content');
            const answerContent = document.querySelector('.answer-content');
            
            const thinkingHTML = thinkingContent.innerHTML;
            const answerHTML = answerContent ? answerContent.innerHTML : '';
            
            if (thinkingHTML.includes('<pre><code>') && !thinkingHTML.includes('</code></pre>')) {
                return '</code></pre>';
            } else if (answerHTML.includes('<pre><code>') && !answerHTML.includes('</code></pre>')) {
                return '</code></pre>';
            } else {
                return '<pre><code>';
            }
        }
        
        return token;
    }

    // Throttle function to limit how often a function can be called
    function throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // Render MathJax for streamed content (throttled)
    const renderMathJax = throttle(function() {
        if (window.MathJax && MathJax.typesetPromise) {
            try {
                // Only typeset visible math that hasn't been processed
                MathJax.typesetPromise([document.querySelector('.ai-result')])
                    .catch(err => {
                        console.error('MathJax error:', err);
                    });
            } catch (e) {
                console.error('Error in MathJax rendering:', e);
            }
        }
    }, 500); // Throttle to max once per 500ms

    // Finalize AI result display
    function finalizeAIResult() {
        // Make sure the answer section is visible
        const answerSection = document.querySelector('.answer-section');
        if (answerSection && answerSection.style.display === 'none') {
            answerSection.style.display = 'block';
        }
        
        // Check for unclosed LaTeX elements in both sections
        const thinkingContent = document.querySelector('.thinking-content');
        const answerContent = document.querySelector('.answer-content');
        
        if (thinkingContent) {
            // Close any unclosed LaTeX elements
            let html = thinkingContent.innerHTML;
            if (html.includes('<div class="latex-equation">$$') && !html.includes('$$</div>')) {
                html += '$$</div>';
                thinkingContent.innerHTML = html;
            }
            if (html.includes('<span class="latex-inline">$') && !html.includes('$</span>')) {
                html += '$</span>';
                thinkingContent.innerHTML = html;
            }
            // Close any unclosed code blocks
            if (html.includes('<pre><code>') && !html.includes('</code></pre>')) {
                html += '</code></pre>';
                thinkingContent.innerHTML = html;
            }
        }
        
        if (answerContent) {
            // Close any unclosed LaTeX elements
            let html = answerContent.innerHTML;
            if (html.includes('<div class="latex-equation">$$') && !html.includes('$$</div>')) {
                html += '$$</div>';
                answerContent.innerHTML = html;
            }
            if (html.includes('<span class="latex-inline">$') && !html.includes('$</span>')) {
                html += '$</span>';
                answerContent.innerHTML = html;
            }
            // Close any unclosed code blocks
            if (html.includes('<pre><code>') && !html.includes('</code></pre>')) {
                html += '</code></pre>';
                answerContent.innerHTML = html;
            }
        }
        
        // Final render of all LaTeX content
        if (window.MathJax && MathJax.typesetPromise) {
            try {
                MathJax.typesetPromise([document.querySelector('.ai-result')]);
            } catch (e) {
                console.error('Error in final MathJax rendering:', e);
            }
        }
    }

    function displayWebResults(results) {
        if (!results || results.length === 0) {
            searchResults.innerHTML = `
                <div class="search-result-item">
                    <div class="search-result-content">No results found</div>
                </div>
            `;
            return;
        }

        searchResults.innerHTML = results.map(result => `
            <div class="search-result-item">
                <div class="search-result-title">
                    <a href="${result.url}" target="_blank">${result.title}</a>
                </div>
                <div class="search-result-content">${result.snippet}</div>
            </div>
        `).join('');
    }

    // Initialize the UI
    switchTab('web');
}); 