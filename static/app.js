document.addEventListener('DOMContentLoaded', function() {
    const urlInput = document.getElementById('urlInput');
    const summarizeBtn = document.getElementById('summarizeBtn');
    const summaryBox = document.getElementById('summaryBox');
    
    if (!urlInput || !summarizeBtn || !summaryBox) {
        console.log('Elements not found - not on index page');
        return;
    }
    
    console.log('SummarAIze app.js loaded successfully');
    
    // FIXED: Use the correct Flask server URL
    const API_URL = 'http://127.0.0.1:5000';
    
    summarizeBtn.addEventListener('click', summarizeWebpage);
    
    urlInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            summarizeWebpage();
        }
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
                headers: {
                    'Content-Type': 'application/json',
                },
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
            <div style="text-align: center; padding: 40px; background: #f9f9f9; border-radius: 8px;">
                <div style="width: 50px; height: 50px; margin: 0 auto; border: 4px solid #f3f3f3; border-top: 4px solid #4CAF50; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                <p style="margin-top: 20px; color: #666;">🔍 Scraping and analyzing content...</p>
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
        
        summaryBox.innerHTML = `
            <div style="background: #f0f9f4; border: 2px solid #4CAF50; border-radius: 10px; padding: 25px;">
                <h3 style="color: #4CAF50; margin: 0 0 15px 0;">✅ Summary Generated</h3>
                
                <div style="background: white; padding: 18px; border-radius: 8px; margin-bottom: 15px;">
                    <div style="margin-bottom: 12px;">
                        <strong>📄 Title:</strong>
                        <p style="margin: 5px 0 0 0;">${escapeHtml(data.title)}</p>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <strong>📊 Content:</strong>
                        <span>${data.content_length.toLocaleString()} characters</span>
                    </div>
                    <div>
                        <strong>🔗 Source:</strong>
                        <a href="${escapeHtml(data.url)}" target="_blank" style="color: #2196F3; word-break: break-all;">${escapeHtml(data.url)}</a>
                    </div>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 8px;">
                    <h4 style="margin: 0 0 15px 0;">📝 Summary:</h4>
                    <p style="line-height: 1.8; margin: 0;">${escapeHtml(data.summary)}</p>
                </div>
            </div>
        `;
        summaryBox.style.display = 'block';
    }
    
    function showError(message) {
        summarizeBtn.disabled = false;
        summarizeBtn.textContent = 'Click to Summarize';
        
        summaryBox.innerHTML = `
            <div style="background: #ffebee; border: 2px solid #f44336; border-radius: 10px; padding: 30px; text-align: center;">
                <div style="font-size: 3em;">❌</div>
                <h3 style="color: #f44336;">Error</h3>
                <p style="color: #d32f2f;">${escapeHtml(message)}</p>
                <button onclick="document.getElementById('summaryBox').style.display='none'" 
                        style="background: #f44336; color: white; border: none; padding: 10px 25px; border-radius: 5px; cursor: pointer;">
                    Close
                </button>
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
        const map = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'};
        return String(text).replace(/[&<>"']/g, m => map[m]);
    }
});