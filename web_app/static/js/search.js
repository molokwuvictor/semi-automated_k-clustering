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
                // Perform web search
                const response = await fetch(`/web_search?query=${encodeURIComponent(query)}`);
                const data = await response.json();
                displayWebResults(data);
            } else {
                // Perform AI search
                const response = await fetch('/ai_search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query })
                });
                const data = await response.json();
                displayAIResults(data);
            }
        } catch (error) {
            searchResults.innerHTML = `
                <div class="search-result-item">
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
        if (!result || !result.response) {
            searchResults.innerHTML = `
                <div class="search-result-item">
                    <div class="search-result-content">No response from AI</div>
                </div>
            `;
            return;
        }

        searchResults.innerHTML = `
            <div class="search-result-item">
                <div class="search-result-content">${result.response}</div>
            </div>
        `;
    }
}); 