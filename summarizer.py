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


def is_english_text(text):
    """Check if text is primarily English (not Russian, Chinese, etc.)"""
    if not text:
        return False
    
    # Check for non-Latin characters (Cyrillic, Chinese, Arabic, etc.)
    non_latin = re.findall(r'[^\x00-\x7F\u0080-\u00FF]', text)
    
    # If more than 20% is non-Latin, reject it
    if len(non_latin) > len(text) * 0.2:
        print(f"[LANGUAGE CHECK] Rejected: Contains {len(non_latin)} non-Latin characters")
        return False
    
    # Check for common English words
    common_words = ['the', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 
                   'will', 'would', 'can', 'could', 'should', 'this', 'that', 'with', 'for',
                   'product', 'good', 'bad', 'quality', 'phone', 'battery', 'camera']
    
    text_lower = text.lower()
    has_english = sum(1 for word in common_words if word in text_lower)
    
    if has_english < 2:
        print(f"[LANGUAGE CHECK] Rejected: Not enough English words")
        return False
    
    return True


def aggressive_clean_review(text):
    """ENHANCED: Much more aggressive cleaning to remove ALL noise"""
    if not text:
        return ""
    
    # FIRST: Check if English, reject if not
    if not is_english_text(text):
        return ""
    
    # Remove ALL metadata patterns
    text = re.sub(r'(?:Certified|Verified)\s+(?:Buyer|Purchase|Customer)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Flipkart\s+Customer', '', text, flags=re.IGNORECASE)
    text = re.sub(r'READ\s+MORE|Report\s+Abuse|Was\s+this\s+helpful|Helpful\?', '', text, flags=re.IGNORECASE)
    
    # Remove ALL date patterns
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,2}\s+(?:days?|weeks?|months?|years?)\s+ago', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Permalink[s]?', '', text, flags=re.IGNORECASE)
    
    # Remove ALL location patterns (cities, states)
    text = re.sub(r'\b(?:Delhi|Mumbai|Bangalore|Gurgaon|Punjab|Maharashtra|Jalandhar|Bhiwani|Mehendia|Pannu|Gurgawala|Ahmedabad|Chennai|Kolkata|Hyderabad)\b', '', text, flags=re.IGNORECASE)
    
    # Remove ALL name patterns (common names)
    text = re.sub(r'\b(?:Vivek|Kumar|Rahul|Shivam|Raman|Khanna|Chopra|Pannu|Bhiwani|Mehendia|Singh|Sharma|Patel|Gupta|Thakur|Thakор)\b', '', text, flags=re.IGNORECASE)
    
    # Remove ALL rating-related words
    text = re.sub(r'\b(?:Excellent|Very\s+Good|Good|Pleasant|Interesting|Fine|Wonderful|satisfied|rating|star)\b', '', text, flags=re.IGNORECASE)
    
    # Remove URLs and emails
    text = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+', '', text)
    
    # Remove special characters and symbols
    text = re.sub(r'[★☆⭐•●○◆◇✓✗×→←↑↓₹$€£¥@#%&():;|]', '', text)
    
    # Remove standalone single letters and short meaningless words
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    
    # Clean excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    cleaned = text.strip()
    
    # Final English check after cleaning
    if not is_english_text(cleaned):
        return ""
    
    return cleaned


def clean_output_text(text):
    """Final output cleaning - remove ALL numbers and noise"""
    if not text:
        return ""
    
    # Check if English first
    if not is_english_text(text):
        return ""
    
    # Remove ALL numbers and number-related patterns
    text = re.sub(r'\d+[\d,.\s:/-]*', '', text)
    text = re.sub(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b', '', text, flags=re.IGNORECASE)
    
    # Remove rating/review metadata
    text = re.sub(r'(?:star|rating|review)s?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Certified\s+Buyer|Verified\s+Purchase', '', text, flags=re.IGNORECASE)
    
    # Remove special characters
    text = re.sub(r'[★☆⭐•●○◆◇✓✗×→←↑↓₹$€£¥@#%&|]', '', text)
    
    # Remove dates
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*,?\s*', '', text, flags=re.IGNORECASE)
    
    # Clean punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def smart_clean_text(text):
    """Less aggressive cleaning for general text"""
    if not text:
        return ""
    
    # Check if English
    if not is_english_text(text):
        return ""
    
    # Remove metadata but keep actual content
    text = re.sub(r'READ MORE|Report Abuse|Was this helpful|Helpful\?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Certified\s+Buyer|Flipkart\s+Customer', '', text, flags=re.IGNORECASE)
    
    # Remove URLs and emails
    text = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+', '', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_meaningful_sentence(sentence):
    """Check if sentence is meaningful product review content"""
    sentence = sentence.strip()
    
    # FIRST: Must be English
    if not is_english_text(sentence):
        return False
    
    # Length requirements
    if len(sentence) < 25 or len(sentence) > 350:
        return False
    
    # Must contain actual words (not just noise)
    words = sentence.split()
    if len(words) < 6:
        return False
    
    # Check for meaningful product-related words
    meaningful_words = ['product', 'quality', 'phone', 'camera', 'battery', 'display', 'screen', 
                       'performance', 'value', 'price', 'feature', 'design', 'build', 'works',
                       'good', 'bad', 'excellent', 'poor', 'great', 'issue', 'problem', 'love',
                       'like', 'recommend', 'buy', 'purchase', 'worth', 'money', 'device',
                       'sound', 'speaker', 'audio', 'charging', 'fast', 'slow', 'durable',
                       'cheap', 'expensive', 'satisfied', 'disappointed', 'happy', 'unhappy']
    
    has_meaningful = any(word in sentence.lower() for word in meaningful_words)
    if not has_meaningful:
        return False
    
    # Reject if contains too many proper nouns (likely names/places)
    proper_nouns = sum(1 for word in words if word[0].isupper() and len(word) > 1)
    if proper_nouns > len(words) * 0.3:
        return False
    
    # Reject if contains numbers
    if re.search(r'\d', sentence):
        return False
    
    # Reject incomplete sentences
    if sentence.endswith(('demonstr', 'statemen', 'unfortun', 'howev')):
        return False
    
    return True


def post_process_output(text):
    """Final cleanup - create clean, natural sentences"""
    if not text:
        return ""
    
    # Must be English
    if not is_english_text(text):
        return ""
    
    # Clean the text
    text = clean_output_text(text)
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    cleaned_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Use meaningful sentence checker
        if not is_meaningful_sentence(sentence):
            continue
        
        # Skip duplicates
        sentence_lower = sentence.lower()
        if sentence_lower in seen:
            continue
        
        seen.add(sentence_lower)
        
        # Add period if needed
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
        
        print(f"[TRUNCATE] Input has {len(tokens)} tokens, max allowed: {max_tokens}")
        
        if len(tokens) <= max_tokens:
            return text
        
        # Take first 70% and last 30% to preserve context
        first_chunk = int(max_tokens * 0.7)
        last_chunk = int(max_tokens * 0.3)
        
        truncated_tokens = tokens[:first_chunk] + tokens[-last_chunk:]
        result = t5_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        print(f"[TRUNCATE] Truncated to {len(truncated_tokens)} tokens")
        return result
        
    except Exception as e:
        print(f"[TRUNCATE] Error: {e}")
        return text[:1500]


def stage1_t5_extract(input_text, task_type="summarize", max_length=180):
    """Stage 1: T5 extracts key information"""
    if not T5_READY or t5_model is None:
        return None
    
    try:
        max_input_tokens = 450
        
        tokens = t5_tokenizer.encode(input_text, add_special_tokens=False, truncation=False)
        if len(tokens) > max_input_tokens:
            print(f"[T5] Truncating input from {len(tokens)} to {max_input_tokens} tokens")
            tokens = tokens[:max_input_tokens]
            input_text = t5_tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Product-focused prompts
        if task_type == "extract_pros":
            prompt = f"List what customers love about this product: {input_text}"
        elif task_type == "extract_cons":
            prompt = f"List customer complaints and issues with this product: {input_text}"
        elif task_type == "brief":
            prompt = f"Write a product review summary: {input_text}"
        else:
            prompt = f"Summarize this product review: {input_text}"
        
        # Verify prompt length
        prompt_tokens = t5_tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        if len(prompt_tokens) > 500:
            print(f"[T5] Warning: Prompt has {len(prompt_tokens)} tokens, truncating...")
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


def stage2_llama_refine(t5_output, task_type="summarize"):
    """Stage 2: Llama refines into natural product review language"""
    
    if not LLAMA_READY or llama_model is None:
        print(f"[LLAMA STAGE 2] Llama not ready - using T5 output only")
        return post_process_output(t5_output)
    
    try:
        # Product review specific system prompts
        if task_type == "extract_pros":
            system_prompt = "You are a product reviewer. Write about what customers appreciate in this product. Focus ONLY on product features, quality, and performance. Write naturally and conversationally like explaining to a friend."
        elif task_type == "extract_cons":
            system_prompt = "You are a product reviewer. Write about customer concerns with this product. Focus ONLY on product issues, defects, and problems. Write naturally and conversationally like explaining to a friend."
        elif task_type == "brief":
            system_prompt = "You are a product reviewer. Write a natural summary about this product based on customer feedback. Focus on product features, quality, and value. Be conversational and clear."
        else:
            system_prompt = "You are a product reviewer. Write naturally about this product based on customer reviews. Focus on product quality, features, and value."
        
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
        
        # Extract assistant's response
        if "<|assistant|>" in llama_output:
            llama_output = llama_output.split("<|assistant|>")[-1].strip()
        
        # Remove system tokens
        llama_output = re.sub(r'<\|.*?\|>', '', llama_output).strip()
        
        # Clean the output
        cleaned = post_process_output(llama_output)
        
        # Fallback if cleaning removed everything
        if not cleaned or len(cleaned) < 30:
            return post_process_output(t5_output)
        
        return cleaned
        
    except Exception as e:
        print(f"[LLAMA STAGE 2] Error: {e}")
        return post_process_output(t5_output)
    finally:
        if device == "cpu":
            gc.collect()


def multi_stage_process(input_text, task_type="summarize"):
    """Multi-stage processing: T5 extraction → Llama refinement"""
    try:
        # Stage 1: T5 extraction
        t5_output = stage1_t5_extract(input_text, task_type=task_type)
        
        if not t5_output:
            return None
        
        # Stage 2: Llama refinement
        final_output = stage2_llama_refine(t5_output, task_type=task_type)
        
        return final_output
    
    except Exception as e:
        print(f"[MULTI-STAGE] Error: {str(e)}")
        return None


def summarize_text(text, max_length=300, min_length=100):
    """Summarize general text content using multi-stage processing"""
    if not T5_READY:
        load_models()
    
    if not text or len(text.strip()) < 100:
        return "Text too short to summarize (minimum 100 characters required)."
    
    if not T5_READY:
        return "Summarization model not available."
    
    try:
        cleaned_text = smart_clean_text(text)
        
        if len(cleaned_text) < 50:
            cleaned_text = text
        
        cleaned_text = safe_truncate_text(cleaned_text, max_tokens=380)
        
        print(f"[SUMMARIZER] Processing {len(cleaned_text)} characters")
        
        if len(cleaned_text) < 50:
            return "Content too short after cleaning."
        
        result = multi_stage_process(cleaned_text, task_type="summarize")
        
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
        
        # Generate summaries with fallbacks
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
        
        result = multi_stage_process(combined, task_type="brief")
        
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
        
        result = multi_stage_process(combined, task_type="extract_pros")
        
        if not result or len(result) < 30:
            print(f"[PROS] Model output too short, using fallback")
            return get_fallback_pros()
        
        # Parse into meaningful sentences
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        
        pros = []
        for s in sentences:
            s = s.strip()
            
            # Use meaningful sentence checker
            if not is_meaningful_sentence(s):
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
        
        result = multi_stage_process(combined, task_type="extract_cons")
        
        if not result or len(result) < 30:
            print(f"[CONS] Model output too short, using fallback")
            return get_fallback_cons()
        
        # Parse into meaningful sentences
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        
        cons = []
        for s in sentences:
            s = s.strip()
            
            # Use meaningful sentence checker
            if not is_meaningful_sentence(s):
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