from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from scraper import scrape_webpage
from summarizer import summarize_text, summarize_reviews_with_analysis, load_model
import time

app = Flask(__name__)
CORS(app)

# Pre-load the model when server starts
print("\n🚀 Starting server and loading ML model...")
load_model()
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
    """Enhanced endpoint for webpage and product review summarization."""
    try:
        data = request.json
        url = data.get('url')

        if not url:
            return jsonify({
                'success': False,
                'error': 'No URL provided'
            }), 400

        # Check cache (valid for 1 hour)
        cache_key = url
        if cache_key in cache:
            cached_data = cache[cache_key]
            if time.time() - cached_data['timestamp'] < 3600:
                print(f"[CACHE] Returning cached result for {url}")
                return jsonify(cached_data['data'])

        # Scrape webpage (with reviews if it's a product page)
        print(f"[API] Scraping URL: {url}")
        scrape_result = scrape_webpage(url, scrape_reviews=True)
        
        if not scrape_result['success']:
            return jsonify({
                'success': False,
                'error': scrape_result['error']
            }), 400

        # Check if this is a product page with reviews
        is_product_page = scrape_result.get('is_product_page', False)
        reviews = scrape_result.get('reviews', [])
        
        print(f"[API] Product page: {is_product_page}, Reviews found: {len(reviews)}")

        # Prepare response based on page type
        if is_product_page and len(reviews) > 0:
            # Product page with reviews - analyze reviews
            print(f"[API] Analyzing {len(reviews)} reviews...")
            review_analysis = summarize_reviews_with_analysis(reviews)
            
            response_data = {
                'success': True,
                'type': 'product',
                'title': scrape_result['title'],
                'url': url,
                'is_product_page': True,
                'review_count': len(reviews),
                'brief_summary': review_analysis.get('brief_summary', ''),
                'detailed_summary': review_analysis.get('detailed_summary', ''),
                'pros': review_analysis.get('pros', []),
                'cons': review_analysis.get('cons', []),
                'stats': review_analysis.get('stats', {}),
                'metadata': scrape_result.get('metadata', {})
            }
        elif is_product_page and len(reviews) == 0:
            # Product page detected but no reviews found
            content = scrape_result['content']
            print(f"[API] Product page but no reviews found. Summarizing product description...")
            
            if len(content) > 100:
                summary = summarize_text(content, max_length=250, min_length=100)
            else:
                summary = "Product page detected but no reviews found. Try accessing the 'All Reviews' page directly."
            
            response_data = {
                'success': True,
                'type': 'product_no_reviews',
                'title': scrape_result['title'],
                'summary': summary,
                'url': url,
                'is_product_page': True,
                'review_count': 0,
                'message': 'Product detected but reviews not found on this page. For Flipkart, try clicking "View All Reviews" and use that URL.',
                'metadata': scrape_result.get('metadata', {})
            }
        else:
            # Regular webpage - summarize content
            content = scrape_result['content']
            if len(content) < 100:
                return jsonify({
                    'success': False,
                    'error': 'Content too short to summarize (less than 100 characters)'
                }), 400

            print(f"[API] Summarizing webpage content ({len(content)} chars)...")
            summary = summarize_text(content, max_length=350, min_length=150)

            response_data = {
                'success': True,
                'type': 'webpage',
                'title': scrape_result['title'],
                'summary': summary,
                'content_length': len(content),
                'url': url,
                'is_product_page': False,
                'metadata': scrape_result.get('metadata', {})
            }

        # Cache result
        cache[cache_key] = {
            'data': response_data,
            'timestamp': time.time()
        }

        print(f"[API] ✅ Successfully processed: {response_data['type']}")
        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True}), 200

if __name__ == '__main__':
    # Disabled reloader to prevent crashes from TensorFlow file changes
    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)