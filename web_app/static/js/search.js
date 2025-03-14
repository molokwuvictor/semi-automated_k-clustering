document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const searchSidebar = document.querySelector('.search-sidebar');
    const searchToggle = document.querySelector('.search-toggle');
    const closeSidebar = document.querySelector('.close-sidebar');
    const searchTabs = document.querySelectorAll('.search-tab');
    const searchInput = document.querySelector('.search-input');
    const searchResults = document.querySelector('.search-results');
    const container = document.querySelector('.container');

    // State
    let currentTab = 'web';
    let searchTimeout = null;

    // Event Listeners
    searchToggle.addEventListener('click', toggleSidebar);
    closeSidebar.addEventListener('click', toggleSidebar);
    searchTabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
    searchInput.addEventListener('input', handleSearchInput);

    // Functions
    function toggleSidebar() {
        searchSidebar.classList.toggle('active');
        container.classList.toggle('sidebar-active');
        if (searchSidebar.classList.contains('active')) {
            searchInput.focus();
        }
    }

    function switchTab(tab) {
        currentTab = tab;
        searchTabs.forEach(t => t.classList.remove('active'));
        document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
        searchResults.innerHTML = ''; // Clear results when switching tabs
        
        // Update placeholder based on tab
        if (tab === 'web') {
            searchInput.placeholder = "Search the web...";
        } else {
            searchInput.placeholder = "Ask DeepSeek R1 AI...";
        }
    }

    function handleSearchInput(e) {
        const query = e.target.value.trim();
        
        // Clear previous timeout
        if (searchTimeout) {
            clearTimeout(searchTimeout);
        }

        // Set new timeout for debouncing
        searchTimeout = setTimeout(() => {
            if (query.length > 0) {
                performSearch(query);
            } else {
                searchResults.innerHTML = '';
            }
        }, 500);
    }

    async function performSearch(query) {
        try {
            searchResults.innerHTML = '<div class="search-result-item"><div class="search-result-content">Searching...</div></div>';
            
            if (currentTab === 'web') {
                // Perform web search using Google API
                const response = await fetch(`/web_search?query=${encodeURIComponent(query)}`);
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Failed to perform web search');
                }
                
                const data = await response.json();
                displayWebResults(data);
            } else {
                // Perform AI search using DeepSeek R1
                const response = await fetch('/ai_search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query })
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Failed to perform AI search');
                }
                
                const data = await response.json();
                displayAIResults(data);
            }
        } catch (error) {
            console.error("Search error:", error);
            searchResults.innerHTML = `
                <div class="search-result-item error">
                    <div class="search-result-content">Error performing search: ${error.message}</div>
                </div>
            `;
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

    function displayAIResults(result) {
        if (!result) {
            searchResults.innerHTML = `
                <div class="search-result-item">
                    <div class="search-result-content">No response from AI</div>
                </div>
            `;
            return;
        }

        // Create the HTML structure
        let resultHTML = '<div class="search-result-item ai-result">';
        
        // Display thinking process if available
        if (result.thinking && result.thinking.trim() !== '') {
            resultHTML += `
                <div class="thinking-section">
                    <div class="thinking-header">
                        <i class="fas fa-brain"></i> Thinking Process
                    </div>
                    <div class="thinking-content">
                        ${formatAIText(result.thinking)}
                    </div>
                </div>
            `;
        }
        
        // Display the final answer
        if (result.response) {
            resultHTML += `
                <div class="answer-section">
                    <div class="answer-header">
                        <i class="fas fa-comment-dots"></i> Answer
                    </div>
                    <div class="answer-content">
                        ${formatAIText(result.response)}
                    </div>
                </div>
            `;
        } else if (result.full_response) {
            // Fallback to full response if response isn't split
            resultHTML += `
                <div class="answer-section">
                    <div class="answer-content">
                        ${formatAIText(result.full_response)}
                    </div>
                </div>
            `;
        }
        
        resultHTML += '</div>';
        searchResults.innerHTML = resultHTML;
    }
    
    function formatAIText(text) {
        if (!text) return '';
        
        // Replace newlines with <br> tags
        let formatted = text.replace(/\n/g, '<br>');
        
        // Format code blocks if they exist (using ```code``` format)
        formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Format inline code if it exists (using `code` format)
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        return formatted;
    }
}); 