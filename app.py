from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from scraper import scrape_webpage
from summarizer import summarize_text, load_model
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
    """Main endpoint for webpage summarization."""
    try:
        data = request.json
        url = data.get('url')

        if not url:
            return jsonify({
                'success': False,
                'error': 'No URL provided'
            }), 400

        # Check cache (valid for 1 hour)
        if url in cache:
            cached_data = cache[url]
            if time.time() - cached_data['timestamp'] < 3600:
                return jsonify(cached_data['data'])

        # Scrape webpage
        scrape_result = scrape_webpage(url)
        if not scrape_result['success']:
            return jsonify({
                'success': False,
                'error': scrape_result['error']
            }), 400

        content = scrape_result['content']
        if len(content) < 100:
            return jsonify({
                'success': False,
                'error': 'Content too short to summarize (less than 100 characters)'
            }), 400

        # Generate summary
        summary = summarize_text(content, max_length=200, min_length=50)

        # Prepare response
        response_data = {
            'success': True,
            'title': scrape_result['title'],
            'summary': summary,
            'content_length': len(content),
            'url': url,
            'metadata': scrape_result.get('metadata', {})
        }

        # Cache result
        cache[url] = {
            'data': response_data,
            'timestamp': time.time()
        }

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
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # FIXED: Disabled reloader to prevent crashes from TensorFlow file changes
    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)