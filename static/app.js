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
                // Check if it's a product page with reviews or regular webpage
                if (data.type === 'product') {
                    showProductReviewAnalysis(data);
                } else if (data.type === 'product_no_reviews') {
                    showProductNoReviews(data);
                } else {
                    showWebpageSummary(data);
                }
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
        summarizeBtn.textContent = 'Analyzing...';
        
        summaryBox.innerHTML = `
            <div style="text-align:center;padding:40px;background:#f9fafb;border-radius:12px;border:2px solid #e5e7eb;">
                <div style="width:60px;height:60px;margin:0 auto;border:5px solid #f3f4f6;
                    border-top:5px solid #22c55e;border-radius:50%;animation:spin 1s linear infinite;"></div>
                <p style="margin-top:24px;color:#6b7280;font-size:16px;font-weight:500;">🔍 Scraping and analyzing content...</p>
                <p style="margin-top:8px;color:#9ca3af;font-size:14px;">This may take a few moments</p>
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

    function showProductNoReviews(data) {
        summarizeBtn.disabled = false;
        summarizeBtn.textContent = 'Summarize Another';

        summaryBox.innerHTML = `
            <div style="max-width:900px;margin:0 auto;padding:24px;">
                
                <!-- Warning Header -->
                <div style="background:linear-gradient(135deg,#f59e0b,#d97706);color:white;padding:32px;border-radius:16px;margin-bottom:24px;box-shadow:0 4px 16px rgba(245,158,11,0.3);">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                        <span style="font-size:32px;">⚠️</span>
                        <h2 style="margin:0;font-size:28px;font-weight:700;">Product Page Detected</h2>
                    </div>
                    <p style="margin:8px 0 0 44px;font-size:18px;opacity:0.95;">No reviews found on this page</p>
                </div>

                <!-- Info Box -->
                <div style="background:white;padding:28px;border-radius:12px;border-left:4px solid #f59e0b;box-shadow:0 2px 8px rgba(0,0,0,0.05);margin-bottom:24px;">
                    <h3 style="margin:0 0 16px 0;font-size:20px;color:#92400e;font-weight:600;display:flex;align-items:center;gap:8px;">
                        <span>💡</span> Tip for Better Results
                    </h3>
                    <p style="margin:0 0 12px 0;line-height:1.8;color:#374151;font-size:16px;">
                        ${escapeHtml(data.message || 'Reviews may be loaded dynamically or on a separate page.')}
                    </p>
                    <p style="margin:0;line-height:1.8;color:#6b7280;font-size:15px;">
                        <strong>For Flipkart:</strong> Click on "View All Reviews" or "Ratings & Reviews" button on the product page, then copy that URL.
                    </p>
                </div>

                <!-- Product Summary -->
                <div style="background:white;border:2px solid #cbd5e1;border-radius:12px;padding:28px;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                        <span style="font-size:28px;">📦</span>
                        <h3 style="margin:0;font-size:20px;color:#475569;font-weight:600;">Product Information</h3>
                    </div>
                    <h4 style="margin:0 0 16px 0;font-size:18px;color:#374151;font-weight:500;">${escapeHtml(data.title)}</h4>
                    <p style="margin:0;line-height:1.8;color:#4b5563;font-size:16px;text-align:justify;">${escapeHtml(data.summary)}</p>
                </div>

                <!-- Suggested Amazon Alternative -->
                <div style="background:#eff6ff;padding:20px;border-radius:12px;margin-top:24px;border:2px dashed #3b82f6;">
                    <p style="margin:0;color:#1e40af;font-size:15px;line-height:1.6;">
                        <strong>💡 Tip:</strong> Amazon product pages usually work better for review analysis. Try searching for a similar product on Amazon India.
                    </p>
                </div>

            </div>
        `;
        
        summaryBox.style.display = 'block';
        setTimeout(() => {
            summaryBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }

    function showProductReviewAnalysis(data) {
        summarizeBtn.disabled = false;
        summarizeBtn.textContent = 'Summarize Another';

        const stats = data.stats || {};
        const avgRating = stats.average_rating || 0;
        const totalReviews = stats.total_reviews || 0;
        const posPercentage = stats.positive_percentage || 0;
        const negPercentage = stats.negative_percentage || 0;

        // Generate star rating HTML
        const starRating = generateStarRating(avgRating);

        // Generate rating distribution bars
        const ratingBars = generateRatingBars(stats.rating_distribution || {});

        summaryBox.innerHTML = `
            <div style="max-width:1200px;margin:0 auto;padding:24px;">
                
                <!-- Header Section -->
                <div style="background:linear-gradient(135deg,#22c55e,#16a34a);color:white;padding:32px;border-radius:16px;margin-bottom:24px;box-shadow:0 4px 16px rgba(34,197,94,0.2);">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                        <span style="font-size:32px;">📦</span>
                        <h2 style="margin:0;font-size:28px;font-weight:700;">Product Review Analysis</h2>
                    </div>
                    <p style="margin:8px 0 0 44px;font-size:18px;opacity:0.95;">${escapeHtml(data.title)}</p>
                </div>

                <!-- Stats Overview -->
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px;">
                    
                    <div style="background:white;padding:20px;border-radius:12px;border:2px solid #fbbf24;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                        <div style="font-size:14px;color:#92400e;font-weight:600;margin-bottom:8px;">⭐ Average Rating</div>
                        <div style="font-size:32px;font-weight:700;color:#92400e;margin-bottom:8px;">${avgRating.toFixed(1)}/5</div>
                        <div>${starRating}</div>
                    </div>

                    <div style="background:white;padding:20px;border-radius:12px;border:2px solid #3b82f6;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                        <div style="font-size:14px;color:#1e40af;font-weight:600;margin-bottom:8px;">📊 Total Reviews</div>
                        <div style="font-size:32px;font-weight:700;color:#1e40af;">${totalReviews}</div>
                        <div style="font-size:13px;color:#6b7280;margin-top:4px;">${data.review_count} analyzed</div>
                    </div>

                    <div style="background:white;padding:20px;border-radius:12px;border:2px solid #22c55e;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                        <div style="font-size:14px;color:#065f46;font-weight:600;margin-bottom:8px;">👍 Positive</div>
                        <div style="font-size:32px;font-weight:700;color:#065f46;">${posPercentage.toFixed(0)}%</div>
                        <div style="background:#dcfce7;height:6px;border-radius:3px;margin-top:8px;overflow:hidden;">
                            <div style="background:#22c55e;height:100%;width:${posPercentage}%;"></div>
                        </div>
                    </div>

                    <div style="background:white;padding:20px;border-radius:12px;border:2px solid #ef4444;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                        <div style="font-size:14px;color:#991b1b;font-weight:600;margin-bottom:8px;">👎 Negative</div>
                        <div style="font-size:32px;font-weight:700;color:#991b1b;">${negPercentage.toFixed(0)}%</div>
                        <div style="background:#fee2e2;height:6px;border-radius:3px;margin-top:8px;overflow:hidden;">
                            <div style="background:#ef4444;height:100%;width:${negPercentage}%;"></div>
                        </div>
                    </div>

                </div>

                <!-- Quick Summary -->
                <div style="background:white;padding:24px;border-radius:12px;margin-bottom:24px;border-left:4px solid #8b5cf6;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                    <h3 style="margin:0 0 16px 0;font-size:20px;color:#6b21a8;font-weight:600;display:flex;align-items:center;gap:8px;">
                        <span>💬</span> Quick Summary
                    </h3>
                    <p style="margin:0;line-height:1.8;color:#374151;font-size:16px;">${escapeHtml(data.brief_summary)}</p>
                </div>

                <!-- Pros and Cons -->
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px;">
                    
                    <!-- Pros -->
                    <div style="background:white;padding:24px;border-radius:12px;border:2px solid #22c55e;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                        <h3 style="margin:0 0 16px 0;font-size:20px;color:#065f46;font-weight:600;display:flex;align-items:center;gap:8px;">
                            <span>✅</span> Pros
                        </h3>
                        ${data.pros && data.pros.length > 0 ? `
                            <ul style="margin:0;padding:0;list-style:none;">
                                ${data.pros.map(pro => `
                                    <li style="padding:12px;margin-bottom:8px;background:#f0fdf4;border-radius:8px;border-left:3px solid #22c55e;color:#065f46;font-size:15px;line-height:1.6;">
                                        ${escapeHtml(pro)}
                                    </li>
                                `).join('')}
                            </ul>
                        ` : '<p style="color:#9ca3af;font-style:italic;">No specific pros identified</p>'}
                    </div>

                    <!-- Cons -->
                    <div style="background:white;padding:24px;border-radius:12px;border:2px solid #ef4444;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                        <h3 style="margin:0 0 16px 0;font-size:20px;color:#991b1b;font-weight:600;display:flex;align-items:center;gap:8px;">
                            <span>❌</span> Cons
                        </h3>
                        ${data.cons && data.cons.length > 0 ? `
                            <ul style="margin:0;padding:0;list-style:none;">
                                ${data.cons.map(con => `
                                    <li style="padding:12px;margin-bottom:8px;background:#fef2f2;border-radius:8px;border-left:3px solid #ef4444;color:#991b1b;font-size:15px;line-height:1.6;">
                                        ${escapeHtml(con)}
                                    </li>
                                `).join('')}
                            </ul>
                        ` : '<p style="color:#9ca3af;font-style:italic;">No specific cons identified</p>'}
                    </div>

                </div>

                <!-- Rating Distribution -->
                <div style="background:white;padding:24px;border-radius:12px;border:2px solid #e5e7eb;box-shadow:0 2px 8px rgba(0,0,0,0.05);margin-bottom:24px;">
                    <h3 style="margin:0 0 20px 0;font-size:20px;color:#374151;font-weight:600;display:flex;align-items:center;gap:8px;">
                        <span>📊</span> Rating Distribution
                    </h3>
                    ${ratingBars}
                </div>

                <!-- Detailed Summary (Collapsible) -->
                <div style="background:white;padding:24px;border-radius:12px;border:2px solid #cbd5e1;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                    <details>
                        <summary style="cursor:pointer;font-size:18px;font-weight:600;color:#475569;padding:8px 0;list-style:none;display:flex;align-items:center;gap:8px;">
                            <span style="transition:transform 0.2s;">▶</span> Detailed Analysis
                        </summary>
                        <div style="margin-top:16px;padding-top:16px;border-top:1px solid #e2e8f0;">
                            <p style="margin:0;line-height:1.8;color:#475569;font-size:15px;">${escapeHtml(data.detailed_summary)}</p>
                        </div>
                    </details>
                </div>

            </div>
        `;

        // Add rotation animation for details arrow
        const detailsElements = summaryBox.querySelectorAll('details');
        detailsElements.forEach(details => {
            details.addEventListener('toggle', function() {
                const arrow = this.querySelector('summary span');
                if (this.open) {
                    arrow.style.transform = 'rotate(90deg)';
                } else {
                    arrow.style.transform = 'rotate(0deg)';
                }
            });
        });

        summaryBox.style.display = 'block';
        setTimeout(() => {
            summaryBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }

    function showWebpageSummary(data) {
        summarizeBtn.disabled = false;
        summarizeBtn.textContent = 'Summarize Another';

        summaryBox.innerHTML = `
            <div style="max-width:900px;margin:0 auto;padding:24px;">
                <div style="background:white;border:2px solid #22c55e;border-radius:12px;padding:28px;box-shadow:0 4px 12px rgba(0,0,0,0.08);">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                        <span style="font-size:28px;">📄</span>
                        <h3 style="margin:0;font-size:22px;color:#065f46;font-weight:600;">Webpage Summary</h3>
                    </div>
                    <h4 style="margin:0 0 16px 0;font-size:18px;color:#374151;font-weight:500;">${escapeHtml(data.title)}</h4>
                    <p style="margin:0;line-height:1.8;color:#4b5563;font-size:16px;text-align:justify;">${escapeHtml(data.summary)}</p>
                    <div style="margin-top:20px;padding-top:16px;border-top:1px solid #e5e7eb;font-size:14px;color:#9ca3af;">
                        Content length: ${data.content_length} characters
                    </div>
                </div>
            </div>
        `;
        summaryBox.style.display = 'block';
        setTimeout(() => {
            summaryBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
    
    function showError(message) {
        summarizeBtn.disabled = false;
        summarizeBtn.textContent = 'Try Again';
        
        summaryBox.innerHTML = `
            <div style="max-width:700px;margin:0 auto;padding:24px;">
                <div style="background:#fef2f2;border:2px solid #ef4444;border-radius:12px;padding:24px;box-shadow:0 2px 8px rgba(239,68,68,0.1);">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                        <span style="font-size:32px;">⚠️</span>
                        <h3 style="margin:0;font-size:20px;color:#991b1b;font-weight:600;">Error</h3>
                    </div>
                    <p style="margin:0;color:#7f1d1d;font-size:16px;line-height:1.6;">${escapeHtml(message)}</p>
                </div>
            </div>
        `;
        summaryBox.style.display = 'block';
    }

    function generateStarRating(rating) {
        const fullStars = Math.floor(rating);
        const hasHalfStar = rating % 1 >= 0.5;
        const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);
        
        let stars = '';
        for (let i = 0; i < fullStars; i++) stars += '⭐';
        if (hasHalfStar) stars += '⭐'; // Using full star for simplicity
        for (let i = 0; i < emptyStars; i++) stars += '☆';
        
        return `<span style="font-size:20px;letter-spacing:2px;">${stars}</span>`;
    }

    function generateRatingBars(distribution) {
        const total = Object.values(distribution).reduce((sum, val) => sum + val, 0);
        if (total === 0) return '<p style="color:#9ca3af;">No rating data available</p>';

        const ratings = [
            { label: '5 ⭐', key: '5_star', color: '#22c55e' },
            { label: '4 ⭐', key: '4_star', color: '#84cc16' },
            { label: '3 ⭐', key: '3_star', color: '#fbbf24' },
            { label: '2 ⭐', key: '2_star', color: '#f97316' },
            { label: '1 ⭐', key: '1_star', color: '#ef4444' }
        ];

        return ratings.map(rating => {
            const count = distribution[rating.key] || 0;
            const percentage = total > 0 ? (count / total * 100).toFixed(1) : 0;
            
            return `
                <div style="margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <span style="font-size:14px;font-weight:600;color:#374151;min-width:60px;">${rating.label}</span>
                        <span style="font-size:13px;color:#6b7280;">${count} reviews (${percentage}%)</span>
                    </div>
                    <div style="background:#f3f4f6;height:10px;border-radius:5px;overflow:hidden;">
                        <div style="background:${rating.color};height:100%;width:${percentage}%;transition:width 0.3s ease;"></div>
                    </div>
                </div>
            `;
        }).join('');
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