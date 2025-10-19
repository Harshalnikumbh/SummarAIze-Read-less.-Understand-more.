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
    """Enhanced endpoint with better content validation and error messages."""
    try:
        data = request.json
        url = data.get('url')
        max_pages = data.get('max_pages', 25)

        if not url:
            return jsonify({
                'success': False,
                'error': 'No URL provided'
            }), 400

        # Check cache
        cache_key = f"{url}_{max_pages}"
        if cache_key in cache:
            cached_data = cache[cache_key]
            if time.time() - cached_data['timestamp'] < 3600:
                print(f"[CACHE] Returning cached result for {url}")
                cached_response = cached_data['data'].copy()
                cached_response['cached'] = True
                return jsonify(cached_response)

        print(f"\n{'='*60}")
        print(f"[API] Scraping URL: {url}")
        print(f"[API] Max pages: {max_pages}")
        print(f"{'='*60}")
        
        start_time = time.time()
        scrape_result = scrape_webpage(url, max_pages=max_pages)
        scrape_time = time.time() - start_time
        
        if not scrape_result['success']:
            return jsonify({
                'success': False,
                'error': scrape_result.get('error', 'Failed to scrape webpage')
            }), 400

        url_type = scrape_result.get('url_type', 'webpage')
        reviews = scrape_result.get('reviews', [])
        content = scrape_result.get('content', '')
        
        print(f"[API] URL Type: {url_type}")
        print(f"[API] Reviews: {len(reviews)}")
        print(f"[API] Content length: {len(content)} chars")
        print(f"[API] Scraping time: {scrape_time:.2f}s")

        # IMPROVED: Content validation
        if len(content) < 100 and len(reviews) == 0:
            return jsonify({
                'success': False,
                'error': 'Insufficient content to summarize',
                'message': 'This page contains very little text. Please try a different page with more substantial content.',
                'details': {
                    'content_length': len(content),
                    'url_type': url_type
                }
            }), 400

        # Process based on content type
        if url_type in ['review_page', 'product_page'] and len(reviews) > 0:
            # Reviews available
            print(f"[API] Analyzing {len(reviews)} reviews...")
            analysis_start = time.time()
            
            review_analysis = summarize_reviews_with_analysis(reviews, max_reviews=200)
            
            analysis_time = time.time() - analysis_start
            total_time = scrape_time + analysis_time
            
            print(f"[API] Analysis time: {analysis_time:.2f}s")
            print(f"[API] Total time: {total_time:.2f}s")
            
            if not review_analysis.get('success', False):
                # Fallback to content summary
                print("[API] Review analysis failed, trying content summary...")
                summary = summarize_text(content) if len(content) >= 100 else "Could not analyze reviews."
                
                response_data = {
                    'success': True,
                    'type': 'webpage',
                    'url_type': url_type,
                    'title': scrape_result['title'],
                    'summary': summary,
                    'content_length': len(content),
                    'url': url,
                    'processing_time': f"{total_time:.2f}s",
                    'note': 'Review analysis failed, showing content summary instead.'
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
            # Product page but no reviews
            print(f"[API] Product page with no reviews")
            
            if len(content) < 100:
                return jsonify({
                    'success': False,
                    'error': 'Product page found but no reviewable content',
                    'message': 'Try accessing the "View All Reviews" link directly.',
                    'url_type': url_type
                }), 400
            
            summary = summarize_text(content, max_length=250, min_length=100)
            
            response_data = {
                'success': True,
                'type': 'product_no_reviews',
                'url_type': url_type,
                'title': scrape_result['title'],
                'summary': summary,
                'url': url,
                'review_count': 0,
                'message': 'Product detected but no reviews found. Try the "View All Reviews" page.',
                'metadata': scrape_result.get('metadata', {}),
                'processing_time': f"{scrape_time:.2f}s"
            }
            
        else:
            # Regular webpage
            if len(content) < 100:
                return jsonify({
                    'success': False,
                    'error': 'Content too short to summarize',
                    'message': 'The page content is too brief (less than 100 characters). Please try a page with more substantial text.',
                    'details': {
                        'content_length': len(content),
                        'title': scrape_result['title']
                    }
                }), 400

            print(f"[API] Regular webpage - summarizing {len(content)} chars...")
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

        print(f"[API] ✅ Success: {response_data['type']}")
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
    print("  ✓ Stage 2: TinyLlama human-like refinement")
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
    
    app.run(debug=True, port=8000, host='0.0.0.0', use_reloader=False)