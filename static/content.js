// Content script runs on product pages
// This is optional - can be used for future enhancements

console.log('Review Summarizer: Content script loaded');

// Detect if current page is a product page
function isProductPage() {
    const url = window.location.href;
    
    // Amazon
    if (url.includes('amazon.') && (url.includes('/dp/') || url.includes('/product/'))) {
        return 'amazon';
    }
    
    // Flipkart
    if (url.includes('flipkart.com') && url.includes('/p/')) {
        return 'flipkart';
    }
    
    // Myntra
    if (url.includes('myntra.com') && url.includes('/buy/')) {
        return 'myntra';
    }
    
    return null;
}

// Get product information from page
function getProductInfo() {
    const platform = isProductPage();
    
    if (!platform) return null;
    
    let productData = {
        url: window.location.href,
        platform: platform,
        title: '',
        image: '',
        price: ''
    };
    
    if (platform === 'amazon') {
        // Extract product title
        const titleEl = document.getElementById('productTitle');
        productData.title = titleEl ? titleEl.textContent.trim() : '';
        
        // Extract product image
        const imageEl = document.getElementById('landingImage');
        productData.image = imageEl ? imageEl.src : '';
        
        // Extract price
        const priceEl = document.querySelector('.a-price-whole');
        productData.price = priceEl ? priceEl.textContent.trim() : '';
    }
    
    return productData;
}

// Optional: Add a floating button on product pages
function addFloatingButton() {
    const platform = isProductPage();
    if (!platform) return;
    
    // Check if button already exists
    if (document.getElementById('review-summarizer-btn')) return;
    
    const button = document.createElement('button');
    button.id = 'review-summarizer-btn';
    button.innerHTML = '📊 Summarize Reviews';
    button.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 10000;
        padding: 12px 24px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    `;
    
    button.addEventListener('mouseenter', () => {
        button.style.transform = 'translateY(-2px)';
        button.style.boxShadow = '0 6px 16px rgba(102, 126, 234, 0.6)';
    });
    
    button.addEventListener('mouseleave', () => {
        button.style.transform = 'translateY(0)';
        button.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.4)';
    });
    
    button.addEventListener('click', () => {
        // Send message to open popup
        chrome.runtime.sendMessage({ action: 'openPopup' });
    });
    
    document.body.appendChild(button);
}

// Initialize
if (isProductPage()) {
    console.log('Product page detected:', isProductPage());
    
    // Optional: Add floating button
    // Uncomment to enable
    // addFloatingButton();
    
    // Send product info to background script
    const productInfo = getProductInfo();
    if (productInfo) {
        chrome.runtime.sendMessage({
            action: 'productDetected',
            data: productInfo
        });
    }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getProductInfo') {
        const productInfo = getProductInfo();
        sendResponse({ success: true, data: productInfo });
    }
    
    return true;
});