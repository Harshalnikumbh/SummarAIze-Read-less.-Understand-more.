from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from scraper import scrape_webpage
from summarizer import summarize_text, summarize_reviews_with_analysis, load_models
import time

app = Flask(__name__)
CORS(app)

# Pre-load the models when server starts
print("\n🚀 Starting server and loading ML models...")
load_models()  # Now loads both T5 and Llama
print("✅ Server ready!\n")

# Simple cache to avoid re-scraping
cache = {}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/help', methods=['GET'])
def help_page():
    return render_template('help.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """Enhanced endpoint with MULTI-STAGE processing and 30-page pagination."""
    try:
        data = request.json
        url = data.get('url')
        max_pages = data.get('max_pages', 30)  # INCREASED TO 30 PAGES FOR BETTER QUALITY

        if not url:
            return jsonify({
                'success': False,
                'error': 'No URL provided'
            }), 400

        # Check cache (valid for 1 hour)
        cache_key = f"{url}_{max_pages}"
        if cache_key in cache:
            cached_data = cache[cache_key]
            if time.time() - cached_data['timestamp'] < 3600:
                print(f"[CACHE] Returning cached result for {url}")
                cached_response = cached_data['data'].copy()
                cached_response['cached'] = True
                return jsonify(cached_response)

        # Scrape webpage with increased pagination
        print(f"\n{'='*60}")
        print(f"[API] Scraping URL: {url}")
        print(f"[API] Max pages: {max_pages} (Multi-stage processing enabled)")
        print(f"{'='*60}")
        
        start_time = time.time()
        scrape_result = scrape_webpage(url, max_pages=max_pages)
        scrape_time = time.time() - start_time
        
        if not scrape_result['success']:
            return jsonify({
                'success': False,
                'error': scrape_result.get('error', 'Failed to scrape webpage')
            }), 400

        # Get URL type and reviews
        url_type = scrape_result.get('url_type', 'webpage')
        reviews = scrape_result.get('reviews', [])
        
        print(f"[API] URL Type: {url_type}")
        print(f"[API] Reviews found: {len(reviews)}")
        print(f"[API] Scraping time: {scrape_time:.2f}s")

        # Prepare response based on URL type and review availability
        if url_type in ['review_page', 'product_page'] and len(reviews) > 0:
            # Product/Review page with reviews - full multi-stage analysis
            print(f"[API] Analyzing {len(reviews)} reviews with MULTI-STAGE pipeline (T5→Llama)...")
            analysis_start = time.time()
            
            # Now using 200 reviews with multi-stage processing
            review_analysis = summarize_reviews_with_analysis(reviews, max_reviews=200)
            
            analysis_time = time.time() - analysis_start
            total_time = scrape_time + analysis_time
            
            print(f"[API] Analysis time: {analysis_time:.2f}s")
            print(f"[API] Total time: {total_time:.2f}s")
            
            if not review_analysis.get('success', False):
                # Fallback if analysis fails
                print("[API] Review analysis failed, falling back to content summary")
                content = scrape_result['content']
                summary = summarize_text(content) if content else "Could not analyze reviews."
                
                response_data = {
                    'success': True,
                    'type': 'webpage',
                    'url_type': url_type,
                    'title': scrape_result['title'],
                    'summary': summary,
                    'content_length': len(content),
                    'url': url,
                    'processing_time': f"{total_time:.2f}s"
                }
            else:
                response_data = {
                    'success': True,
                    'type': 'product',
                    'url_type': url_type,
                    'title': scrape_result['title'],
                    'url': url,
                    'review_count': len(reviews),
                    'brief_summary': review_analysis.get('brief_summary', ''),
                    'detailed_summary': review_analysis.get('detailed_summary', ''),
                    'pros': review_analysis.get('pros', []),
                    'cons': review_analysis.get('cons', []),
                    'stats': review_analysis.get('stats', {}),
                    'metadata': scrape_result.get('metadata', {}),
                    'scrape_time': f"{scrape_time:.2f}s",
                    'analysis_time': f"{analysis_time:.2f}s",
                    'processing_time': f"{total_time:.2f}s",
                    'processing_mode': 'multi-stage (T5→Llama)'
                }
                
        elif url_type == 'product_page' and len(reviews) == 0:
            # Product page detected but no reviews found
            content = scrape_result['content']
            print(f"[API] Product page but no reviews found")
            print(f"[API] Summarizing product description ({len(content)} chars)...")
            
            if len(content) > 100:
                summary = summarize_text(content, max_length=250, min_length=100)
            else:
                summary = "Product page detected but no reviews found. Try accessing the 'All Reviews' page directly."
            
            response_data = {
                'success': True,
                'type': 'product_no_reviews',
                'url_type': url_type,
                'title': scrape_result['title'],
                'summary': summary,
                'url': url,
                'review_count': 0,
                'message': 'Product detected but reviews not found on this page. For Flipkart, try clicking "View All Reviews" and use that URL. For Amazon, click "See all reviews".',
                'metadata': scrape_result.get('metadata', {}),
                'processing_time': f"{scrape_time:.2f}s"
            }
            
        else:
            # Regular webpage - summarize content with multi-stage processing
            content = scrape_result['content']
            
            if len(content) < 100:
                return jsonify({
                    'success': False,
                    'error': 'Content too short to summarize (less than 100 characters)'
                }), 400

            print(f"[API] Regular webpage - Multi-stage summarization ({len(content)} chars)...")
            summary_start = time.time()
            
            summary = summarize_text(content, max_length=400, min_length=150)
            
            summary_time = time.time() - summary_start
            total_time = scrape_time + summary_time
            
            print(f"[API] Summary time: {summary_time:.2f}s")
            print(f"[API] Total time: {total_time:.2f}s")

            response_data = {
                'success': True,
                'type': 'webpage',
                'url_type': url_type,
                'title': scrape_result['title'],
                'summary': summary,
                'content_length': len(content),
                'url': url,
                'metadata': scrape_result.get('metadata', {}),
                'scrape_time': f"{scrape_time:.2f}s",
                'summary_time': f"{summary_time:.2f}s",
                'processing_time': f"{total_time:.2f}s",
                'processing_mode': 'multi-stage (T5→Llama)'
            }

        # Cache result
        cache[cache_key] = {
            'data': response_data,
            'timestamp': time.time()
        }

        print(f"[API] ✅ Successfully processed: {response_data['type']}")
        print(f"{'='*60}\n")
        return jsonify(response_data)

    except Exception as e:
        import traceback
        print(f"\n[API] ❌ ERROR:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    from summarizer import T5_READY, LLAMA_READY
    return jsonify({
        'status': 'healthy',
        't5_loaded': T5_READY,
        'llama_loaded': LLAMA_READY,
        'processing_mode': 'multi-stage' if (T5_READY and LLAMA_READY) else 't5-only' if T5_READY else 'unavailable',
        'cache_size': len(cache)
    }), 200

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the cache."""
    global cache
    cache_size = len(cache)
    cache = {}
    return jsonify({
        'success': True,
        'message': f'Cleared {cache_size} cached items'
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SummarAIze Flask Server Starting...")
    print("="*60)
    print("Features:")
    print("  ✓ MULTI-STAGE PROCESSING (T5 → Llama)")
    print("  ✓ Enhanced pagination (30 pages)")
    print("  ✓ Stage 1: FLAN-T5 extraction")
    print("  ✓ Stage 2: Llama human-like refinement")
    print("  ✓ Analyzes up to 200 reviews per product")
    print("  ✓ Ultra-aggressive text cleaning")
    print("  ✓ Intelligent URL detection")
    print("  ✓ Smart caching (1 hour)")
    print("="*60)
    print("Endpoints:")
    print("  POST /summarize    - Main endpoint")
    print("  GET  /health       - Health check (shows T5/Llama status)")
    print("  POST /clear-cache  - Clear cache")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)