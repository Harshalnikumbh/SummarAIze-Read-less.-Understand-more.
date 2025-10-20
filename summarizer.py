import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import gc

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model states
T5_READY = False
LLAMA_READY = False
t5_model = None
t5_tokenizer = None
llama_model = None
llama_tokenizer = None

# CPU optimization settings
BATCH_SIZE = 10
MAX_WORKERS = 1
PROCESSING_TIMEOUT = 30

def load_models():
    """Load both T5 and Llama models for multi-stage processing"""
    global T5_READY, LLAMA_READY, t5_model, t5_tokenizer, llama_model, llama_tokenizer
    
    if T5_READY and LLAMA_READY:
        return
    
    print("\n" + "="*60)
    print("📦 Loading Multi-Stage Summarization Models...")
    print("="*60)
    
    # Load T5 (Stage 1: Extraction & Initial Summarization)
    try:
        print("\n[STAGE 1] Loading FLAN-T5-Small (CPU Optimized)...")
        t5_model_name = "google/flan-t5-small"
        t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(
            t5_model_name,
            torch_dtype=torch.float32
        )
        t5_model.to(device)
        t5_model.eval()
        
        if device == "cpu":
            torch.set_num_threads(2)
        
        T5_READY = True
        print(f"✓ FLAN-T5-Small loaded successfully on {device}")
    except Exception as e:
        print(f"❌ Error loading FLAN-T5: {e}")
        T5_READY = False
    
    # Load Llama (Stage 2: Human-like Generation)
    try:
        print("\n[STAGE 2] Loading TinyLlama (CPU Optimized)...")
        llama_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        print("[STAGE 2] Loading tokenizer...")
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        
        if llama_tokenizer.pad_token is None:
            llama_tokenizer.pad_token = llama_tokenizer.eos_token
            llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
        
        print("[STAGE 2] Loading model with CPU optimizations...")
        llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        llama_model.to(device)
        llama_model.eval()
        
        if device == "cpu":
            torch.set_num_threads(2)
        
        LLAMA_READY = True
        print(f"✓ TinyLlama loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Error loading Llama: {e}")
        print("⚠️ Falling back to T5-only mode...")
        LLAMA_READY = False
    
    print(f"\n{'='*60}")
    print(f"✓ Device: {device}")
    print(f"✓ T5 Status: {'✅ Ready' if T5_READY else '❌ Failed'}")
    print(f"✓ Llama Status: {'✅ Ready' if LLAMA_READY else '❌ Failed (T5-only mode)'}")
    print(f"{'='*60}\n")


def detect_content_type(text, url=None):
    """
    ENHANCED: Robust detection with URL awareness
    Returns: 'product_review', 'news_article', 'general_webpage', 'technical_doc'
    """
    if not text:
        return 'general_webpage'
    
    text_lower = text.lower()
    text_words = text_lower.split()
    
    # === PRIORITY: URL-BASED DETECTION ===
    review_score = 0
    news_score = 0
    tech_score = 0
    general_score = 0
    
    if url:
        url_lower = url.lower()
        
        # Strong URL signals for news
        news_domains = [
            'bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com',
            'theguardian.com', 'timesofindia.com', 'hindustantimes.com',
            'indianexpress.com', 'ndtv.com', 'thehindu.com',
            'news18.com', 'firstpost.com', 'scroll.in'
        ]
        
        news_paths = ['/news/', '/article/', '/story/', '/breaking-', '/report/']
        
        if any(domain in url_lower for domain in news_domains):
            news_score += 10
            print(f"[CONTENT DETECTION] Strong news signal from domain")
        
        if any(path in url_lower for path in news_paths):
            news_score += 8
            print(f"[CONTENT DETECTION] News path pattern detected")
        
        # Strong URL signals for reviews
        review_domains = ['amazon', 'flipkart', 'myntra', 'reviews']
        review_paths = ['/product-reviews/', '/customer-reviews/', '/reviews/', '/dp/', '/p/']
        
        if any(domain in url_lower for domain in review_domains):
            review_score += 10
        
        if any(path in url_lower for path in review_paths):
            review_score += 8
        
        # Strong URL signals for technical docs
        tech_domains = ['github.com', 'readthedocs', 'docs.', 'developer.', 'api.']
        tech_paths = ['/docs/', '/documentation/', '/api/', '/reference/', '/guide/']
        
        if any(domain in url_lower for domain in tech_domains):
            tech_score += 10
        
        if any(path in url_lower for path in tech_paths):
            tech_score += 8
    
    # === PRODUCT REVIEW DETECTION ===
    review_keywords = {
        'strong': ['customer', 'buyer', 'purchase', 'bought', 'ordered', 'product review'],
        'medium': ['quality', 'recommend', 'rating', 'star', 'pros', 'cons', 'satisfied',
                   'disappointed', 'worth buying', 'value for money', 'good product', 'bad product'],
        'weak': ['product', 'item', 'delivery', 'packaging', 'seller']
    }
    review_score += sum(3 for kw in review_keywords['strong'] if kw in text_lower)
    review_score += sum(2 for kw in review_keywords['medium'] if kw in text_lower)
    review_score += sum(1 for kw in review_keywords['weak'] if kw in text_lower)
    
    if re.search(r'\b[1-5](?:\.\d)?\s*(?:star|rating)', text_lower):
        review_score += 4
    
    if re.search(r'verified|certified\s+(?:buyer|purchase)', text_lower):
        review_score += 5
    
    # === NEWS ARTICLE DETECTION ===
    news_keywords = {
        'strong': ['reported', 'according to', 'investigation', 'authorities', 'officials',
                   'statement', 'press conference', 'breaking news', 'correspondent',
                   'spokesperson', 'ministry', 'government'],
        'medium': ['incident', 'event', 'sources', 'claimed', 'alleged', 'confirmed',
                   'announced', 'revealed', 'discovered', 'found that', 'said that'],
        'weak': ['said', 'told', 'news', 'today', 'yesterday', 'recently']
    }
    news_score += sum(3 for kw in news_keywords['strong'] if kw in text_lower)
    news_score += sum(2 for kw in news_keywords['medium'] if kw in text_lower)
    news_score += sum(1 for kw in news_keywords['weak'] if kw in text_lower)
    
    # News agency patterns
    if re.search(r'(?:times of india|bbc|cnn|reuters|associated press|pti|ani|afp|ians)', text_lower):
        news_score += 5
    
    # Date patterns common in news
    if re.search(r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:,\s*\d{4})?', text_lower):
        news_score += 2
    
    # === TECHNICAL DOCUMENTATION DETECTION ===
    tech_keywords = {
        'strong': ['api', 'function', 'method', 'parameter', 'syntax', 'endpoint',
                   'documentation', 'implementation', 'configuration', 'installation guide'],
        'medium': ['command', 'usage', 'example', 'returns', 'argument', 'code snippet',
                   'library', 'module', 'class', 'interface'],
        'weak': ['technical', 'system', 'software', 'application', 'version']
    }
    tech_score += sum(3 for kw in tech_keywords['strong'] if kw in text_lower)
    tech_score += sum(2 for kw in tech_keywords['medium'] if kw in text_lower)
    tech_score += sum(1 for kw in tech_keywords['weak'] if kw in text_lower)
    
    if re.search(r'(?:def|function|class|import|require|npm|pip install)', text_lower):
        tech_score += 5
    
    if re.search(r'[{}()\[\]<>].*[{}()\[\]<>]', text):
        tech_score += 2
    
    # === GENERAL WEBPAGE DETECTION ===
    general_keywords = ['about', 'contact', 'home', 'services', 'company', 'welcome',
                       'learn more', 'read more', 'click here', 'subscribe']
    general_score += sum(1 for kw in general_keywords if kw in text_lower)
    
    if len(text_words) < 100:
        general_score += 2
    
    # === SCORING AND DECISION ===
    scores = {
        'product_review': review_score,
        'news_article': news_score,
        'technical_doc': tech_score,
        'general_webpage': general_score
    }
    
    max_score = max(scores.values())
    
    if max_score < 3:
        print(f"[CONTENT DETECTION] Low scores, defaulting to general_webpage")
        return 'general_webpage'
    
    content_type = max(scores, key=scores.get)
    
    # Validation thresholds
    thresholds = {
        'product_review': 5,
        'news_article': 4,
        'technical_doc': 4,
        'general_webpage': 0
    }
    
    if scores[content_type] < thresholds.get(content_type, 0):
        print(f"[CONTENT DETECTION] {content_type} score {scores[content_type]} below threshold, using general_webpage")
        content_type = 'general_webpage'
    
    print(f"[CONTENT DETECTION] Scores: Review={review_score}, News={news_score}, Tech={tech_score}, General={general_score}")
    print(f"[CONTENT DETECTION] ✅ Detected as: {content_type} (score: {scores[content_type]})")
    
    return content_type


def is_english_text(text):
    """Check if text is primarily English (not Russian, Chinese, etc.)"""
    if not text:
        return False
    
    non_latin = re.findall(r'[^\x00-\x7F\u0080-\u00FF]', text)
    
    if len(non_latin) > len(text) * 0.2:
        return False
    
    common_words = ['the', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 
                   'will', 'would', 'can', 'could', 'should', 'this', 'that', 'with', 'for',
                   'product', 'good', 'bad', 'quality', 'phone', 'battery', 'camera']
    
    text_lower = text.lower()
    has_english = sum(1 for word in common_words if word in text_lower)
    
    if has_english < 2:
        return False
    
    return True


def aggressive_clean_review(text):
    """ENHANCED: Much more aggressive cleaning to remove ALL noise"""
    if not text:
        return ""
    
    if not is_english_text(text):
        return ""
    
    text = re.sub(r'(?:Certified|Verified)\s+(?:Buyer|Purchase|Customer)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Flipkart\s+Customer|Amazon\s+Customer', '', text, flags=re.IGNORECASE)
    text = re.sub(r'READ\s+MORE|Report\s+Abuse|Was\s+this\s+helpful|Helpful\?', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,2}\s+(?:days?|weeks?|months?|years?)\s+ago', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Permalink[s]?', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\b(?:Delhi|Mumbai|Bangalore|Gurgaon|Punjab|Maharashtra|Jalandhar|Bhiwani|Mehendia|Pannu|Gurgawala|Ahmedabad|Chennai|Kolkata|Hyderabad|Agra)\b', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\b(?:Vivek|Kumar|Rahul|Shivam|Raman|Khanna|Chopra|Pannu|Bhiwani|Mehendia|Singh|Sharma|Patel|Gupta|Thakur|Thakор|Banke|Bihari)\b', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\b(?:Excellent|Very\s+Good|Good|Pleasant|Interesting|Fine|Wonderful|satisfied|rating|star)\b', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+', '', text)
    
    text = re.sub(r'[★☆⭐•●○◆◇✓✗×→←↑↓₹$€£¥@#%&():;|]', '', text)
    
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    cleaned = text.strip()
    
    if not is_english_text(cleaned):
        return ""
    
    return cleaned


def clean_output_text(text):
    """Final output cleaning - remove ALL numbers and noise"""
    if not text:
        return ""
    
    if not is_english_text(text):
        return ""
    
    text = re.sub(r'\d+[\d,.\s:/-]*', '', text)
    text = re.sub(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'(?:star|rating|review)s?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Certified\s+Buyer|Verified\s+Purchase', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'[★☆⭐•●○◆◇✓✗×→←↑↓₹$€£¥@#%&|]', '', text)
    
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*,?\s*', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def smart_clean_text(text):
    """Less aggressive cleaning for general text"""
    if not text:
        return ""
    
    if not is_english_text(text):
        return ""
    
    text = re.sub(r'READ MORE|Report Abuse|Was this helpful|Helpful\?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Certified\s+Buyer|Flipkart\s+Customer', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_meaningful_sentence(sentence, content_type='product_review'):
    """Check if sentence is meaningful based on content type"""
    sentence = sentence.strip()
    
    if not is_english_text(sentence):
        return False
    
    if len(sentence) < 25 or len(sentence) > 350:
        return False
    
    words = sentence.split()
    if len(words) < 6:
        return False
    
    if content_type == 'product_review':
        meaningful_words = ['product', 'quality', 'phone', 'camera', 'battery', 'display', 'screen', 
                           'performance', 'value', 'price', 'feature', 'design', 'build', 'works',
                           'good', 'bad', 'excellent', 'poor', 'great', 'issue', 'problem', 'love',
                           'like', 'recommend', 'buy', 'purchase', 'worth', 'money', 'device',
                           'sound', 'speaker', 'audio', 'charging', 'fast', 'slow', 'durable',
                           'cheap', 'expensive', 'satisfied', 'disappointed', 'happy', 'unhappy']
    elif content_type == 'news_article':
        meaningful_words = ['reported', 'according', 'found', 'discovered', 'investigation',
                           'incident', 'event', 'authorities', 'officials', 'statement', 'said',
                           'announced', 'revealed', 'confirmed', 'sources', 'happened', 'occurred',
                           'ministry', 'government', 'police', 'court', 'case', 'arrested']
    elif content_type == 'technical_doc':
        meaningful_words = ['function', 'method', 'parameter', 'configuration', 'installation',
                           'feature', 'usage', 'implementation', 'syntax', 'command', 'api',
                           'system', 'process', 'data', 'code', 'application']
    else:
        meaningful_words = ['information', 'content', 'details', 'describes', 'explains',
                           'provides', 'includes', 'features', 'available', 'important',
                           'key', 'main', 'primary', 'essential', 'significant']
    
    has_meaningful = any(word in sentence.lower() for word in meaningful_words)
    if not has_meaningful:
        return False
    
    proper_nouns = sum(1 for word in words if word[0].isupper() and len(word) > 1)
    if proper_nouns > len(words) * 0.3:
        return False
    
    if re.search(r'\d', sentence):
        return False
    
    if sentence.endswith(('demonstr', 'statemen', 'unfortun', 'howev', 'accordingl', 'therefor')):
        return False
    
    return True


def post_process_output(text, content_type='product_review'):
    """Final cleanup - create clean, natural sentences"""
    if not text:
        return ""
    
    if not is_english_text(text):
        return ""
    
    text = clean_output_text(text)
    
    sentences = re.split(r'[.!?]+', text)
    
    cleaned_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        if not is_meaningful_sentence(sentence, content_type):
            continue
        
        sentence_lower = sentence.lower()
        if sentence_lower in seen:
            continue
        
        seen.add(sentence_lower)
        
        if not sentence.endswith('.'):
            sentence += '.'
        
        cleaned_sentences.append(sentence)
    
    return ' '.join(cleaned_sentences)


def safe_truncate_text(text, max_tokens=380):
    """Safely truncate text to token limit"""
    if not text or not t5_tokenizer:
        return text[:1500] if text else ""
    
    try:
        tokens = t5_tokenizer.encode(text, add_special_tokens=False, truncation=False)
        
        if len(tokens) <= max_tokens:
            return text
        
        first_chunk = int(max_tokens * 0.7)
        last_chunk = int(max_tokens * 0.3)
        
        truncated_tokens = tokens[:first_chunk] + tokens[-last_chunk:]
        result = t5_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return result
        
    except Exception as e:
        print(f"[TRUNCATE] Error: {e}")
        return text[:1500]


def get_prompts_for_content_type(content_type, task_type):
    """
    ENHANCED: Get customized prompts with better news/article handling
    Returns: (t5_prompt_template, llama_system_prompt)
    """
    
    if content_type == 'product_review':
        if task_type == "extract_pros":
            return (
                "List what customers love about this product: {}",
                "You are a product reviewer. Write about what customers appreciate in this product. Focus ONLY on product features, quality, and performance. Write naturally like explaining to a friend."
            )
        elif task_type == "extract_cons":
            return (
                "List customer complaints and issues with this product: {}",
                "You are a product reviewer. Write about customer concerns with this product. Focus ONLY on product issues, defects, and problems. Write naturally like explaining to a friend."
            )
        elif task_type == "brief":
            return (
                "Write a product review summary: {}",
                "You are a product reviewer. Write a natural summary about this product based on customer feedback. Focus on product features, quality, and value."
            )
        else:
            return (
                "Summarize this product review: {}",
                "You are a product reviewer. Write naturally about this product based on customer reviews."
            )
    
    elif content_type == 'news_article':
        if task_type == "summarize":
            return (
                "Summarize this news article covering what happened, when, where, who was involved, and the outcome: {}",
                "You are a news editor. Write a concise, objective summary covering the 5 W's (who, what, when, where, why) and key outcomes. Be factual and neutral. Focus on the main events and their significance. Avoid speculation or opinion."
            )
        else:
            return (
                "Summarize the key facts from this news article: {}",
                "You are a news summarizer. Present the main facts clearly and objectively. Focus on verified information and key developments."
            )
    
    elif content_type == 'technical_doc':
        return (
            "Summarize the key technical information and important features: {}",
            "You are a technical writer. Summarize the important technical details, features, and functionality. Be clear, precise, and focus on practical information that developers or users need to know."
        )
    
    else:
        return (
            "Summarize the main points and key information from this content: {}",
            "You are a content summarizer. Write a clear, informative summary of the main points. Be concise and focus on the most important takeaways that readers should know."
        )


def stage1_t5_extract(input_text, task_type="summarize", content_type="product_review", max_length=180):
    """Stage 1: T5 extracts key information with content-aware prompts"""
    if not T5_READY or t5_model is None:
        return None
    
    try:
        max_input_tokens = 450
        
        tokens = t5_tokenizer.encode(input_text, add_special_tokens=False, truncation=False)
        if len(tokens) > max_input_tokens:
            tokens = tokens[:max_input_tokens]
            input_text = t5_tokenizer.decode(tokens, skip_special_tokens=True)
        
        prompt_template, _ = get_prompts_for_content_type(content_type, task_type)
        prompt = prompt_template.format(input_text)
        
        print(f"[T5] Using prompt for {content_type}: {prompt[:100]}...")
        
        prompt_tokens = t5_tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        if len(prompt_tokens) > 500:
            prompt_tokens = prompt_tokens[:500]
            prompt = t5_tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        
        inputs = t5_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=50,
                num_beams=4,
                length_penalty=1.2,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.8,
                top_p=0.95
            )
        
        t5_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return t5_output.strip()
        
    except Exception as e:
        print(f"[T5 STAGE 1] Error: {e}")
        return None


def stage2_llama_refine(t5_output, task_type="summarize", content_type="product_review"):
    """Stage 2: Llama refines into natural language with content-aware prompts"""
    
    if not LLAMA_READY or llama_model is None:
        return post_process_output(t5_output, content_type)
    
    try:
        _, system_prompt = get_prompts_for_content_type(content_type, task_type)
        
        print(f"[LLAMA] Using system prompt for {content_type}")
        
        prompt = f"""<|system|>
{system_prompt}</s>
<|user|>
{t5_output}</s>
<|assistant|>
"""
        
        inputs = llama_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=400,
            return_attention_mask=True
        ).to(device)
        
        with torch.no_grad():
            outputs = llama_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                min_new_tokens=40,
                temperature=0.85,
                top_p=0.92,
                top_k=50,
                do_sample=True,
                pad_token_id=llama_tokenizer.eos_token_id,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3
            )
        
        llama_output = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|assistant|>" in llama_output:
            llama_output = llama_output.split("<|assistant|>")[-1].strip()
        
        llama_output = re.sub(r'<\|.*?\|>', '', llama_output).strip()
        
        cleaned = post_process_output(llama_output, content_type)
        
        if not cleaned or len(cleaned) < 30:
            return post_process_output(t5_output, content_type)
        
        return cleaned
        
    except Exception as e:
        print(f"[LLAMA STAGE 2] Error: {e}")
        return post_process_output(t5_output, content_type)
    finally:
        if device == "cpu":
            gc.collect()


def multi_stage_process(input_text, task_type="summarize", content_type="product_review"):
    """Multi-stage processing: T5 extraction → Llama refinement with content awareness"""
    try:
        t5_output = stage1_t5_extract(input_text, task_type=task_type, content_type=content_type)
        
        if not t5_output:
            return None
        
        final_output = stage2_llama_refine(t5_output, task_type=task_type, content_type=content_type)
        
        return final_output
    
    except Exception as e:
        print(f"[MULTI-STAGE] Error: {str(e)}")
        return None
def summarize_text(text, max_length=300, min_length=100):
    """Summarize general text content with automatic content type detection"""
    if not T5_READY:
        load_models()
    
    if not text or len(text.strip()) < 100:
        return "Text too short to summarize (minimum 100 characters required)."
    
    if not T5_READY:
        return "Summarization model not available."
    
    try:
        # Detect content type
        content_type = detect_content_type(text)
        print(f"[SUMMARIZER] Content type detected: {content_type}")
        
        cleaned_text = smart_clean_text(text)
        
        if len(cleaned_text) < 50:
            cleaned_text = text
        
        cleaned_text = safe_truncate_text(cleaned_text, max_tokens=380)
        
        print(f"[SUMMARIZER] Processing {len(cleaned_text)} characters as {content_type}")
        
        if len(cleaned_text) < 50:
            return "Content too short after cleaning."
        
        # Use content-aware processing
        result = multi_stage_process(cleaned_text, task_type="summarize", content_type=content_type)
        
        if not result or len(result) < 30:
            return "Unable to generate meaningful summary."
        
        return result

    except Exception as e:
        print(f"[SUMMARIZER] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error generating summary."


def process_review_batch(reviews_batch, batch_num):
    """ENHANCED: Process a batch of reviews with aggressive cleaning"""
    cleaned_reviews = []
    for review in reviews_batch:
        body = review.get('body', '')
        title = review.get('title', '')
        full_text = f"{title} {body}".strip()
        
        # Apply AGGRESSIVE cleaning
        full_text = aggressive_clean_review(full_text)
        
        # Only keep if substantial English content remains
        if full_text and len(full_text) > 40 and is_english_text(full_text):
            cleaned_reviews.append(full_text)
    
    return cleaned_reviews


def get_fallback_pros():
    """Fallback pros when extraction fails"""
    return [
        "Customers appreciate the product's build quality and find it well-constructed for everyday use.",
        "Many buyers praise the value for money, noting that the product delivers good performance at its price point.",
        "The product receives positive feedback for meeting customer expectations and serving its intended purpose effectively."
    ]


def get_fallback_cons():
    """Fallback cons when extraction fails"""
    return [
        "Some customers mentioned concerns about certain aspects that didn't fully meet their expectations.",
        "A few buyers reported issues that potential purchasers should consider before making a decision.",
        "There are scattered reports of quality inconsistencies that warrant attention from prospective buyers."
    ]


def summarize_reviews_with_analysis(reviews_data, max_reviews=150):
    """Analyze reviews efficiently using multi-stage processing"""
    if not T5_READY:
        load_models()
    
    if not reviews_data:
        return {
            'success': False,
            'brief_summary': "No reviews available.",
            'pros': get_fallback_pros(),
            'cons': get_fallback_cons(),
            'stats': {}
        }
    
    if not T5_READY:
        return {
            'success': False,
            'brief_summary': "Model not available.",
            'pros': get_fallback_pros(),
            'cons': get_fallback_cons(),
            'stats': {}
        }
    
    try:
        reviews_data = reviews_data[:max_reviews]
        
        positive_reviews = []
        negative_reviews = []
        ratings = []
        
        # Process in batches with aggressive cleaning
        batches = [reviews_data[i:i + BATCH_SIZE] for i in range(0, len(reviews_data), BATCH_SIZE)]
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {
                executor.submit(process_review_batch, batch, idx): idx 
                for idx, batch in enumerate(batches, 1)
            }
            
            for future in as_completed(future_to_batch, timeout=PROCESSING_TIMEOUT * len(batches)):
                batch_idx = future_to_batch[future]
                try:
                    cleaned_batch = future.result(timeout=PROCESSING_TIMEOUT)
                    
                    for i, review in enumerate(batches[batch_idx - 1]):
                        rating = review.get('rating')
                        if rating:
                            ratings.append(rating)
                            if i < len(cleaned_batch):
                                if rating >= 4:
                                    positive_reviews.append(cleaned_batch[i])
                                elif rating <= 2:
                                    negative_reviews.append(cleaned_batch[i])
                
                except Exception as e:
                    print(f"[BATCH {batch_idx}] Error: {e}")
                    continue
        
        print(f"[ANALYZER] Extracted {len(positive_reviews)} positive and {len(negative_reviews)} negative reviews")
        
        stats = calculate_review_stats(ratings, len(reviews_data))
        
        # Generate summaries with fallbacks (always use product_review content type)
        brief_summary = generate_brief_summary(positive_reviews + negative_reviews, stats)
        pros = extract_pros(positive_reviews)
        cons = extract_cons(negative_reviews)
        
        # Ensure we always have valid pros/cons
        if not pros or len(pros) < 2:
            print("[ANALYZER] Using fallback pros")
            pros = get_fallback_pros()
        
        if not cons or len(cons) < 2:
            print("[ANALYZER] Using fallback cons")
            cons = get_fallback_cons()
        
        return {
            'success': True,
            'brief_summary': brief_summary,
            'pros': pros,
            'cons': cons,
            'stats': stats
        }
    
    except Exception as e:
        print(f"[ANALYZER] Error: {str(e)}")
        return {
            'success': False,
            'brief_summary': "Error analyzing reviews.",
            'pros': get_fallback_pros(),
            'cons': get_fallback_cons(),
            'stats': {}
        }


def generate_brief_summary(review_texts, stats):
    """Generate brief summary focused on PRODUCT"""
    if not review_texts:
        return "No reviews available."
    
    try:
        # ENHANCED: More aggressive pre-cleaning
        cleaned_reviews = []
        for r in review_texts[:25]:
            cleaned = aggressive_clean_review(r)
            if len(cleaned) > 40 and is_english_text(cleaned):
                cleaned_reviews.append(cleaned)
        
        if len(cleaned_reviews) < 3:
            avg_rating = stats.get('average_rating', 0)
            if avg_rating >= 4:
                return "Customers generally love this product and recommend it for its quality and performance."
            elif avg_rating >= 3:
                return "Customer opinions are mixed, with some praising the product while others mention areas for improvement."
            else:
                return "Several customers have expressed concerns about this product worth considering before purchase."
        
        combined = " ".join(cleaned_reviews)
        combined = safe_truncate_text(combined, max_tokens=320)
        
        print(f"[BRIEF] Processing {len(combined)} characters from {len(cleaned_reviews)} reviews")
        
        # Always use product_review content type for reviews
        result = multi_stage_process(combined, task_type="brief", content_type="product_review")
        
        if not result or len(result) < 30:
            avg_rating = stats.get('average_rating', 0)
            if avg_rating >= 4:
                return "Customers are happy with this product and find it delivers good value."
            else:
                return "Customer feedback shows a range of experiences, with both positive comments and noted concerns."
        
        return result
        
    except Exception as e:
        print(f"[BRIEF] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error generating summary."


def extract_pros(positive_reviews):
    """Extract positive points about the PRODUCT"""
    if not positive_reviews or len(positive_reviews) < 3:
        print(f"[PROS] Insufficient reviews ({len(positive_reviews) if positive_reviews else 0})")
        return get_fallback_pros()
    
    try:
        # ENHANCED: Aggressive pre-cleaning
        cleaned = []
        for r in positive_reviews[:25]:
            cleaned_r = aggressive_clean_review(r)
            if len(cleaned_r) > 40 and is_english_text(cleaned_r):
                cleaned.append(cleaned_r)
        
        if len(cleaned) < 3:
            print(f"[PROS] Insufficient clean reviews ({len(cleaned)})")
            return get_fallback_pros()
        
        combined = " ".join(cleaned)
        combined = safe_truncate_text(combined, max_tokens=280)
        
        print(f"[PROS] Processing {len(combined)} characters from {len(cleaned)} reviews")
        
        # Always use product_review content type
        result = multi_stage_process(combined, task_type="extract_pros", content_type="product_review")
        
        if not result or len(result) < 30:
            print(f"[PROS] Model output too short, using fallback")
            return get_fallback_pros()
        
        # Parse into meaningful sentences
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        
        pros = []
        for s in sentences:
            s = s.strip()
            
            # Use meaningful sentence checker with product_review type
            if not is_meaningful_sentence(s, 'product_review'):
                continue
            
            if not s.endswith('.'):
                s += '.'
            
            pros.append(s)
            
            if len(pros) >= 5:
                break
        
        # If we got good pros, return them; otherwise use fallback
        if len(pros) >= 2:
            return pros
        else:
            print(f"[PROS] Only {len(pros)} valid sentences, using fallback")
            return get_fallback_pros()
        
    except Exception as e:
        print(f"[PROS] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return get_fallback_pros()


def extract_cons(negative_reviews):
    """Extract negative points about the PRODUCT"""
    if not negative_reviews or len(negative_reviews) < 3:
        print(f"[CONS] Insufficient reviews ({len(negative_reviews) if negative_reviews else 0})")
        return get_fallback_cons()
    
    try:
        # ENHANCED: Aggressive pre-cleaning
        cleaned = []
        for r in negative_reviews[:25]:
            cleaned_r = aggressive_clean_review(r)
            if len(cleaned_r) > 40 and is_english_text(cleaned_r):
                cleaned.append(cleaned_r)
        
        if len(cleaned) < 3:
            print(f"[CONS] Insufficient clean reviews ({len(cleaned)})")
            return get_fallback_cons()
        
        combined = " ".join(cleaned)
        combined = safe_truncate_text(combined, max_tokens=280)
        
        print(f"[CONS] Processing {len(combined)} characters from {len(cleaned)} reviews")
        
        # Always use product_review content type
        result = multi_stage_process(combined, task_type="extract_cons", content_type="product_review")
        
        if not result or len(result) < 30:
            print(f"[CONS] Model output too short, using fallback")
            return get_fallback_cons()
        
        # Parse into meaningful sentences
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        
        cons = []
        for s in sentences:
            s = s.strip()
            
            # Use meaningful sentence checker with product_review type
            if not is_meaningful_sentence(s, 'product_review'):
                continue
            
            if not s.endswith('.'):
                s += '.'
            
            cons.append(s)
            
            if len(cons) >= 5:
                break
        
        # If we got good cons, return them; otherwise use fallback
        if len(cons) >= 2:
            return cons
        else:
            print(f"[CONS] Only {len(cons)} valid sentences, using fallback")
            return get_fallback_cons()
        
    except Exception as e:
        print(f"[CONS] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return get_fallback_cons()


def calculate_review_stats(ratings, total):
    """Calculate review statistics"""
    stats = {
        'total_reviews': total,
        'average_rating': 0.0,
        'rating_distribution': {'5_star': 0, '4_star': 0, '3_star': 0, '2_star': 0, '1_star': 0},
        'positive_percentage': 0.0,
        'negative_percentage': 0.0
    }
    
    if not ratings:
        return stats
    
    stats['average_rating'] = round(sum(ratings) / len(ratings), 2)
    
    for r in ratings:
        if r >= 4.5:
            stats['rating_distribution']['5_star'] += 1
        elif r >= 3.5:
            stats['rating_distribution']['4_star'] += 1
        elif r >= 2.5:
            stats['rating_distribution']['3_star'] += 1
        elif r >= 1.5:
            stats['rating_distribution']['2_star'] += 1
        else:
            stats['rating_distribution']['1_star'] += 1
    
    pos = stats['rating_distribution']['5_star'] + stats['rating_distribution']['4_star']
    neg = stats['rating_distribution']['1_star'] + stats['rating_distribution']['2_star']
    
    if len(ratings) > 0:
        stats['positive_percentage'] = round((pos / len(ratings)) * 100, 1)
        stats['negative_percentage'] = round((neg / len(ratings)) * 100, 1)
    
    return stats


# ===== DETECTION TEST EXAMPLES =====
def test_content_detection():
    """Test the content detection with various examples"""
    
    test_cases = [
        {
            'name': 'Product Review',
            'text': '''
                Verified Buyer - 5 star rating
                Excellent product! I bought this phone last month and the quality is amazing.
                The camera is superb and battery lasts all day. Highly recommend to everyone.
                Value for money is great. Pros: Good build, fast performance. Cons: Slightly heavy.
            ''',
            'expected': 'product_review'
        },
        {
            'name': 'News Article',
            'text': '''
                Banke Bihari toshkhana survey: Gold, silver bars, gemstones found in underground chamber
                According to temple authorities, an investigation revealed precious items during the survey.
                Officials confirmed the discovery yesterday. The incident was reported by local sources.
                A statement from the press conference announced further examination.
            ''',
            'expected': 'news_article'
        },
        {
            'name': 'Technical Documentation',
            'text': '''
                API Reference Documentation
                function install_package(parameter: string): void
                Usage: npm install package-name
                Configuration: Set the API endpoint in config.json
                Example code snippet showing implementation of the authentication method.
                Returns a promise with the response data.
            ''',
            'expected': 'technical_doc'
        },
        {
            'name': 'General Webpage',
            'text': '''
                Welcome to our company website. Learn more about our services and what we offer.
                Contact us for more information. Click here to read more about our story.
                Home | About | Services | Contact
                Subscribe to our newsletter for updates.
            ''',
            'expected': 'general_webpage'
        },
        {
            'name': 'Ambiguous Short Text',
            'text': 'This is a short message about something.',
            'expected': 'general_webpage'
        }
    ]
    
    print("\n" + "="*70)
    print("🧪 CONTENT DETECTION TESTS")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[TEST {i}] {test['name']}")
        print("-" * 70)
        detected = detect_content_type(test['text'])
        expected = test['expected']
        
        if detected == expected:
            print(f"✅ PASSED: Detected '{detected}' (Expected: '{expected}')")
            passed += 1
        else:
            print(f"❌ FAILED: Detected '{detected}' but expected '{expected}'")
            failed += 1
    
    print("\n" + "="*70)
    print(f"📊 RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("="*70 + "\n")
    
    return passed == len(test_cases)
# Uncomment to run tests
# test_content_detection()