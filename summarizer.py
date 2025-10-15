import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import re

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model states
T5_READY = False
LLAMA_READY = False
t5_model = None
t5_tokenizer = None
llama_model = None
llama_tokenizer = None

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
        print("\n[STAGE 1] Loading FLAN-T5 (Extraction Model)...")
        t5_model_name = "google/flan-t5-base"
        t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)
        t5_model.to(device)
        t5_model.eval()
        T5_READY = True
        print(f"✓ FLAN-T5 loaded successfully")
    except Exception as e:
        print(f"❌ Error loading FLAN-T5: {e}")
        T5_READY = False
    
    # Load Llama (Stage 2: Human-like Generation)
    try:
        print("\n[STAGE 2] Loading Llama (Generation Model)...")
        # Using smaller Llama variant that can run locally
        llama_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        llama_model.to(device)
        llama_model.eval()
        LLAMA_READY = True
        print(f"✓ Llama loaded successfully")
    except Exception as e:
        print(f"❌ Error loading Llama: {e}")
        print("Falling back to T5-only mode...")
        LLAMA_READY = False
    
    print(f"\n✓ Device: {device}")
    print(f"✓ T5 Status: {'Ready' if T5_READY else 'Failed'}")
    print(f"✓ Llama Status: {'Ready' if LLAMA_READY else 'Failed (T5-only mode)'}")
    print("="*60 + "\n")


def stage1_t5_extract(input_text, task_type="summarize", max_length=200):
    """Stage 1: T5 extracts key information and creates initial summary"""
    if not T5_READY or t5_model is None:
        return None
    
    try:
        # Construct prompt based on task type
        if task_type == "extract_pros":
            prompt = "List the positive aspects mentioned in these reviews"
        elif task_type == "extract_cons":
            prompt = "List the negative aspects and issues mentioned in these reviews"
        elif task_type == "brief":
            prompt = "Summarize the main points from these customer reviews in 2-3 sentences"
        elif task_type == "detailed":
            prompt = "Provide a detailed analysis of these customer reviews"
        else:
            prompt = "Summarize this content clearly and concisely"
        
        full_prompt = f"{prompt}: {input_text}"
        
        inputs = t5_tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=max(30, max_length // 4),
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7
            )
        
        t5_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return t5_output.strip()
        
    except Exception as e:
        print(f"[T5 STAGE 1] Error: {e}")
        return None


def stage2_llama_refine(t5_output, task_type="summarize"):
    """Stage 2: Llama refines T5 output into natural, human-like text"""
    if not LLAMA_READY or llama_model is None:
        # Fallback: return T5 output if Llama not available
        return t5_output
    
    try:
        # Create Llama prompt based on task type
        if task_type == "extract_pros":
            system_prompt = "You are a helpful assistant that rewrites review summaries in natural language. Convert the following points into clear, concise bullet points about product benefits."
        elif task_type == "extract_cons":
            system_prompt = "You are a helpful assistant that rewrites review summaries in natural language. Convert the following points into clear, concise bullet points about product issues."
        elif task_type == "brief":
            system_prompt = "You are a helpful assistant that creates clear, natural summaries. Rewrite the following summary to be more readable and natural, keeping it brief (2-3 sentences)."
        elif task_type == "detailed":
            system_prompt = "You are a helpful assistant that creates comprehensive summaries. Expand and refine the following summary into a detailed, natural analysis."
        else:
            system_prompt = "You are a helpful assistant that improves text clarity. Rewrite the following text to be more natural and readable."
        
        # Format for TinyLlama chat template
        prompt = f"""<|system|>
{system_prompt}</s>
<|user|>
{t5_output}</s>
<|assistant|>
"""
        
        inputs = llama_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = llama_model.generate(
                inputs.input_ids,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=llama_tokenizer.eos_token_id
            )
        
        llama_output = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in llama_output:
            llama_output = llama_output.split("<|assistant|>")[-1].strip()
        
        return llama_output.strip()
        
    except Exception as e:
        print(f"[LLAMA STAGE 2] Error: {e}")
        return t5_output  # Fallback to T5 output


def multi_stage_process(input_text, task_type="summarize"):
    """Multi-stage processing: T5 extraction → Llama refinement"""
    print(f"[MULTI-STAGE] Starting {task_type} processing...")
    
    # Stage 1: T5 extraction
    t5_output = stage1_t5_extract(input_text, task_type=task_type)
    if not t5_output:
        return None
    
    print(f"[STAGE 1 OUTPUT] {t5_output[:100]}...")
    
    # Stage 2: Llama refinement
    final_output = stage2_llama_refine(t5_output, task_type=task_type)
    
    print(f"[STAGE 2 OUTPUT] {final_output[:100]}...")
    
    return final_output


# ============== Enhanced Text Cleaning ==============

def aggressive_clean_text(text):
    """Ultra aggressive cleaning to remove ALL noise"""
    text = re.sub(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s*\d{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', '', text)
    text = re.sub(r'\d+\s+(?:month|day|year|week)s?\s+ago', '', text, flags=re.IGNORECASE)
    text = re.sub(r',\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?=\s|$)', '', text)
    text = re.sub(r'(?:Flipkart|Amazon|Myntra)\s+Customer', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Certified\s+Buyer', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?=\w)', '', text)
    text = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+', '', text)
    text = re.sub(r'READ MORE|Report Abuse|Was this helpful|👍|👎', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def safe_truncate_text(text, max_tokens=450):
    """Safely truncate text to fit token limit"""
    if not t5_tokenizer:
        return text[:max_tokens * 4]
    
    try:
        tokens = t5_tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if len(tokens) <= max_tokens:
            return text
        
        chunk1 = tokens[:int(max_tokens * 0.6)]
        chunk2 = tokens[-int(max_tokens * 0.4):]
        combined = chunk1 + chunk2
        return t5_tokenizer.decode(combined, skip_special_tokens=True)
    except:
        return text[:max_tokens * 4]


# ============== Review Analysis Functions ==============

def summarize_reviews_with_analysis(reviews_data, max_reviews=200):
    """Enhanced review analysis with multi-stage processing (INCREASED to 200 reviews)"""
    if not T5_READY:
        load_models()
    
    if not reviews_data:
        return {
            'success': False,
            'brief_summary': "No reviews available to analyze.",
            'detailed_summary': "No reviews available to analyze.",
            'pros': [],
            'cons': [],
            'stats': {}
        }
    
    if not T5_READY:
        return {
            'success': False,
            'brief_summary': "Summarization model not available.",
            'detailed_summary': "Summarization model not available.",
            'pros': [],
            'cons': [],
            'stats': {}
        }
    
    try:
        print(f"\n[REVIEW ANALYZER] Processing {len(reviews_data)} reviews with multi-stage pipeline")
        
        reviews_data = reviews_data[:max_reviews]
        
        # Separate and clean reviews
        positive_reviews = []
        negative_reviews = []
        ratings = []
        
        for review in reviews_data:
            rating = review.get('rating')
            body = review.get('body', '')
            title = review.get('title', '')
            full_text = f"{title} {body}".strip()
            full_text = aggressive_clean_text(full_text)
            
            if not full_text or len(full_text) < 30:
                continue
            
            if rating:
                ratings.append(rating)
                if rating >= 4:
                    positive_reviews.append(full_text)
                elif rating <= 2:
                    negative_reviews.append(full_text)
        
        print(f"[REVIEW ANALYZER] Cleaned - Positive: {len(positive_reviews)}, Negative: {len(negative_reviews)}")
        
        stats = calculate_review_stats(ratings, len(reviews_data))
        
        # Multi-stage processing for each component
        brief_summary = generate_brief_summary_multistage(positive_reviews + negative_reviews, stats)
        detailed_summary = generate_detailed_summary_multistage(positive_reviews + negative_reviews)
        pros = extract_pros_multistage(positive_reviews)
        cons = extract_cons_multistage(negative_reviews)
        
        return {
            'success': True,
            'brief_summary': brief_summary,
            'detailed_summary': detailed_summary,
            'pros': pros,
            'cons': cons,
            'stats': stats
        }
    
    except Exception as e:
        print(f"[REVIEW ANALYZER] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'brief_summary': "Error analyzing reviews.",
            'detailed_summary': "Error analyzing reviews.",
            'pros': [],
            'cons': [],
            'stats': {}
        }


def generate_brief_summary_multistage(review_texts, stats):
    """Generate brief summary using multi-stage processing"""
    if not review_texts:
        return "No reviews available."
    
    try:
        # Prepare input (use more reviews now - 75 instead of 50)
        cleaned_reviews = [aggressive_clean_text(r) for r in review_texts[:75]]
        combined = " ".join([r for r in cleaned_reviews if len(r) > 30])
        combined = safe_truncate_text(combined, max_tokens=450)
        
        if len(combined) < 50:
            return "Insufficient review content for summary."
        
        # Multi-stage processing
        result = multi_stage_process(combined, task_type="brief")
        
        if not result or len(result) < 30:
            return "Could not generate summary."
        
        # Add rating context
        if stats.get('average_rating'):
            avg = stats['average_rating']
            sentiment = "highly positive" if avg >= 4.5 else "positive" if avg >= 4.0 else "mixed" if avg >= 3.0 else "negative"
            result = f"Overall sentiment is {sentiment} ({avg}/5 stars). {result}"
        
        return result
        
    except Exception as e:
        print(f"[BRIEF MULTI-STAGE] Error: {str(e)}")
        return "Error generating brief summary."


def generate_detailed_summary_multistage(review_texts):
    """Generate detailed summary using multi-stage processing"""
    if not review_texts:
        return "No reviews available."
    
    try:
        # Use more reviews (100 instead of 75)
        cleaned_reviews = [aggressive_clean_text(r) for r in review_texts[:100]]
        combined = " ".join([r for r in cleaned_reviews if len(r) > 30])
        combined = safe_truncate_text(combined, max_tokens=450)
        
        if len(combined) < 50:
            return "Insufficient review content for detailed analysis."
        
        # Multi-stage processing
        result = multi_stage_process(combined, task_type="detailed")
        
        if not result or len(result) < 50:
            return "Could not generate detailed summary."
        
        return result
        
    except Exception as e:
        print(f"[DETAILED MULTI-STAGE] Error: {str(e)}")
        return "Error generating detailed summary."


def extract_pros_multistage(positive_reviews):
    """Extract pros using multi-stage processing"""
    if not positive_reviews or len(positive_reviews) < 3:
        return []
    
    try:
        # Use more reviews (80 instead of 60)
        cleaned = [aggressive_clean_text(r) for r in positive_reviews[:80]]
        combined = " ".join([r for r in cleaned if len(r) > 30])
        combined = safe_truncate_text(combined, max_tokens=400)
        
        if len(combined) < 50:
            return ["Customers are generally satisfied with the product."]
        
        # Multi-stage processing
        result = multi_stage_process(combined, task_type="extract_pros")
        
        if not result:
            return ["Customers are generally satisfied with the product."]
        
        # Parse into list
        pros = []
        sentences = [s.strip() for s in re.split(r'[.\n•-]+', result)]
        for s in sentences:
            if len(s) > 20 and len(s) < 200:
                if not s.endswith('.'):
                    s += '.'
                pros.append(s)
                if len(pros) >= 5:
                    break
        
        return pros if pros else ["Customers are generally satisfied with the product."]
        
    except Exception as e:
        print(f"[PROS MULTI-STAGE] Error: {str(e)}")
        return ["Customers are generally satisfied with the product."]


def extract_cons_multistage(negative_reviews):
    """Extract cons using multi-stage processing"""
    if not negative_reviews or len(negative_reviews) < 3:
        return []
    
    try:
        # Use more reviews (80 instead of 60)
        cleaned = [aggressive_clean_text(r) for r in negative_reviews[:80]]
        combined = " ".join([r for r in cleaned if len(r) > 30])
        combined = safe_truncate_text(combined, max_tokens=400)
        
        if len(combined) < 50:
            return ["Some customers reported issues with the product."]
        
        # Multi-stage processing
        result = multi_stage_process(combined, task_type="extract_cons")
        
        if not result:
            return ["Some customers reported issues with the product."]
        
        # Parse into list
        cons = []
        sentences = [s.strip() for s in re.split(r'[.\n•-]+', result)]
        for s in sentences:
            if len(s) > 20 and len(s) < 200:
                if not s.endswith('.'):
                    s += '.'
                cons.append(s)
                if len(cons) >= 5:
                    break
        
        return cons if cons else ["Some customers reported issues with the product."]
        
    except Exception as e:
        print(f"[CONS MULTI-STAGE] Error: {str(e)}")
        return ["Some customers reported issues with the product."]


def summarize_text(text, max_length=400, min_length=150):
    """Enhanced text summarization with multi-stage processing"""
    if not T5_READY:
        load_models()
    
    if not text or len(text.strip()) < 100:
        return "Text too short to summarize."
    
    if not T5_READY:
        return "Summarization model not available."
    
    try:
        print(f"\n[TEXT SUMMARIZER] Input: {len(text)} characters")
        
        cleaned_text = aggressive_clean_text(text)
        cleaned_text = safe_truncate_text(cleaned_text, max_tokens=450)
        
        if len(cleaned_text.strip()) < 50:
            return "Content too short after cleaning."
        
        # Multi-stage processing
        result = multi_stage_process(cleaned_text, task_type="summarize")
        
        if not result or len(result) < 30:
            return "Could not generate summary."
        
        print(f"[TEXT SUMMARIZER] Generated: {len(result)} chars")
        return result

    except Exception as e:
        print(f"[TEXT SUMMARIZER] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error generating summary."


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
    
    stats['positive_percentage'] = round((pos / len(ratings)) * 100, 1)
    stats['negative_percentage'] = round((neg / len(ratings)) * 100, 1)
    
    return stats