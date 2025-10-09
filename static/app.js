document.addEventListener('DOMContentLoaded', function() {
    const urlInput = document.getElementById('urlInput');
    const summarizeBtn = document.getElementById('summarizeBtn');
    const summaryBox = document.getElementById('summaryBox');
    
    if (!urlInput || !summarizeBtn || !summaryBox) {
        console.log('Elements not found - not on index page');
        return;
    }
    
    console.log('SummarAIze app.js loaded successfully');
    
    const API_URL = 'http://127.0.0.1:5000';
    
    summarizeBtn.addEventListener('click', summarizeWebpage);
    urlInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') summarizeWebpage();
    });
    
    async function summarizeWebpage() {
        const url = urlInput.value.trim();
        console.log('Summarize button clicked, URL:', url);
        
        if (!url) {
            showError('Please enter a URL');
            return;
        }
        
        if (!isValidUrl(url)) {
            showError('Please enter a valid URL (e.g., https://example.com)');
            return;
        }
        
        showLoading();
        
        try {
            console.log('Sending POST request to:', `${API_URL}/summarize`);
            
            const response = await fetch(`${API_URL}/summarize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            });
            
            console.log('Response status:', response.status);
            const data = await response.json();
            console.log('Response data:', data);
            
            if (data.success) {
                showSuccess(data);
            } else {
                showError(data.error || 'Failed to summarize webpage');
            }
        } catch (error) {
            console.error('Fetch error:', error);
            showError('Connection error: ' + error.message);
        }
    }
    
    function showLoading() {
        summarizeBtn.disabled = true;
        summarizeBtn.textContent = 'Summarizing...';
        
        summaryBox.innerHTML = `
            <div style="text-align:center;padding:40px;background:#f9f9f9;border-radius:8px;">
                <div style="width:50px;height:50px;margin:0 auto;border:4px solid #f3f3f3;
                    border-top:4px solid #4CAF50;border-radius:50%;animation:spin 1s linear infinite;"></div>
                <p style="margin-top:20px;color:#666;">🔍 Scraping and analyzing content...</p>
            </div>
        `;
        summaryBox.style.display = 'block';
        
        if (!document.getElementById('spinner-style')) {
            const style = document.createElement('style');
            style.id = 'spinner-style';
            style.textContent = '@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }';
            document.head.appendChild(style);
        }
    }

   function showSuccess(data) {
    summarizeBtn.disabled = false;
    summarizeBtn.textContent = 'Click to Summarize';

    // Wide, compact layout
    summaryBox.innerHTML = `
        <div style="background:#ffffff;border:2px solid #22c55e;border-radius:12px;padding:24px;">
            <h3 style="margin:0 0 12px 0;font-size:20px;color:#065f46;font-weight:600;">${escapeHtml(data.title || 'Summary')}</h3>
            <p style="margin:0;line-height:1.7;color:#374151;white-space:pre-wrap;text-align:justify;">${escapeHtml(data.summary)}</p>
        </div>
    `;
    summaryBox.style.display = 'block';
    
    // Scroll to summary smoothly
    setTimeout(() => {
        summaryBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}
    
    function showError(message) {
        summarizeBtn.disabled = false;
        summarizeBtn.textContent = 'Click to Summarize';
        
        summaryBox.innerHTML = `
    <div class="summary-container">
        <h3 class="summary-title">${escapeHtml(data.title || 'Summary')}</h3>
        <p class="summary-text">${escapeHtml(data.summary)}</p>
    </div>
`;
        summaryBox.style.display = 'block';
    }
    
    function isValidUrl(string) {
        try {
            const url = new URL(string);
            return url.protocol === 'http:' || url.protocol === 'https:';
        } catch (_) {
            return false;
        }
    }
    
    function escapeHtml(text) {
        const map = {
            '&': '&amp;', '<': '&lt;', '>': '&gt;',
            '"': '&quot;', "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, m => map[m]);
    }
});
