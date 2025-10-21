// Background service worker for Chrome Extension
// Handles background tasks and communication

// Listen for extension installation
chrome.runtime.onInstalled.addListener(() => {
    console.log('Review Summarizer Extension Installed');
    
    // Set default settings
    chrome.storage.local.set({
        apiUrl: 'http://localhost:5000',
        cacheEnabled: true,
        cacheExpiry: 3600000 // 1 hour in milliseconds
    });
});

// Listen for messages from popup or content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'analyzeReviews') {
        handleAnalyzeReviews(request.url)
            .then(result => sendResponse({ success: true, data: result }))
            .catch(error => sendResponse({ success: false, error: error.message }));
        
        return true; // Keep message channel open for async response
    }
    
    if (request.action === 'getSettings') {
        chrome.storage.local.get(['apiUrl', 'cacheEnabled'], (settings) => {
            sendResponse(settings);
        });
        return true;
    }
});

// Handle review analysis
async function handleAnalyzeReviews(url) {
    // Check cache first
    const cached = await getCachedResult(url);
    if (cached) {
        console.log('Returning cached result');
        return cached;
    }
    
    // If not cached, this would typically call the API
    // But in our case, popup.js handles the API call directly
    // This is here for future extensions
    return null;
}

// Cache management
async function getCachedResult(url) {
    return new Promise((resolve) => {
        chrome.storage.local.get(['cache'], (result) => {
            const cache = result.cache || {};
            const cached = cache[url];
            
            if (cached && Date.now() - cached.timestamp < 3600000) {
                resolve(cached.data);
            } else {
                resolve(null);
            }
        });
    });
}

async function setCachedResult(url, data) {
    return new Promise((resolve) => {
        chrome.storage.local.get(['cache'], (result) => {
            const cache = result.cache || {};
            cache[url] = {
                data: data,
                timestamp: Date.now()
            };
            
            chrome.storage.local.set({ cache }, () => {
                resolve();
            });
        });
    });
}

// Clear old cache entries (run periodically)
function clearOldCache() {
    chrome.storage.local.get(['cache'], (result) => {
        const cache = result.cache || {};
        const now = Date.now();
        const cleaned = {};
        
        for (const [url, entry] of Object.entries(cache)) {
            if (now - entry.timestamp < 3600000) {
                cleaned[url] = entry;
            }
        }
        
        chrome.storage.local.set({ cache: cleaned });
    });
}

// Run cache cleanup every hour
setInterval(clearOldCache, 3600000);

// Listen for tab updates to detect product pages
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url) {
        // Check if it's a product page
        const isProductPage = 
            (tab.url.includes('amazon.') && (tab.url.includes('/dp/') || tab.url.includes('/product/'))) ||
            (tab.url.includes('flipkart.com') && tab.url.includes('/p/')) ||
            (tab.url.includes('myntra.com') && tab.url.includes('/buy/'));
        
        if (isProductPage) {
            // Could show badge or notification
            chrome.action.setBadgeText({ text: '✓', tabId: tabId });
            chrome.action.setBadgeBackgroundColor({ color: '#4CAF50', tabId: tabId });
        } else {
            chrome.action.setBadgeText({ text: '', tabId: tabId });
        }
    }
});