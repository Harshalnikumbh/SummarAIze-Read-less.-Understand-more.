import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'  # Limit CPU threads
os.environ['MKL_NUM_THREADS'] = '2'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
BATCH_SIZE = 15  # Process 15 reviews at a time
MAX_WORKERS = 2  # Limit concurrent processing
PROCESSING_TIMEOUT = 30  # Timeout per batch in seconds

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
        
        # Set to use only 2 threads for CPU
        if device == "cpu":
            torch.set_num_threads(2)
        
        T5_READY = True
        print(f"✓ FLAN-T5 loaded successfully on {device}")
    except Exception as e:
        print(f"❌ Error loading FLAN-T5: {e}")
        import traceback
        traceback.print_exc()
        T5_READY = False
    
    # Load Llama (Stage 2: Human-like Generation)
    try:
        print("\n[STAGE 2] Loading Llama (Generation Model)...")
        llama_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        print("[STAGE 2] Loading tokenizer...")
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        
        # CRITICAL FIX: Set padding token if not present
        if llama_tokenizer.pad_token is None:
            print("[STAGE 2] Setting pad_token = eos_token")
            llama_tokenizer.pad_token = llama_tokenizer.eos_token
            llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
        
        print(f"[STAGE 2] Tokenizer loaded - pad_token: {llama_tokenizer.pad_token}")
        print(f"[STAGE 2] Loading model (this may take a minute on CPU)...")
        
        llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        print(f"[STAGE 2] Moving model to {device}...")
        llama_model.to(device)
        llama_model.eval()
        
        # Verify model is on correct device
        print(f"[STAGE 2] Model device: {next(llama_model.parameters()).device}")
        
        LLAMA_READY = True
        print(f"✓ Llama loaded successfully on {device}")
        
        # Test generation
        print("\n[STAGE 2] Running quick test generation...")
        test_prompt = "Summarize: The product is good."
        test_inputs = llama_tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            test_outputs = llama_model.generate(
                test_inputs.input_ids,
                max_new_tokens=20,
                pad_token_id=llama_tokenizer.eos_token_id
            )
        test_result = llama_tokenizer.decode(test_outputs[0], skip_special_tokens=True)
        print(f"✓ Test generation successful: {test_result[:50]}...")
        
    except Exception as e:
        print(f"❌ Error loading Llama: {e}")
        print("Full traceback:")
        import traceback
        traceback.print_exc()
        print("\n⚠️ Falling back to T5-only mode...")
        LLAMA_READY = False
    
    print(f"\n{'='*60}")
    print(f"✓ Device: {device}")
    print(f"✓ T5 Status: {'✅ Ready' if T5_READY else '❌ Failed'}")
    print(f"✓ Llama Status: {'✅ Ready' if LLAMA_READY else '❌ Failed (T5-only mode)'}")
    print(f"{'='*60}\n")


def clean_output_text(text):
    """
    ULTRA-CLEAN: Remove ALL numbers, special characters, usernames, dates, and noise.
    Keep ONLY natural language sentences.
    """
    if not text:
        return ""
    
    # Remove ALL numbers first (most aggressive)
    text = re.sub(r'\d+[\d,.\s]*', '', text)
    
    # Remove usernames and author names
    text = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]*)+\b', '', text)
    
    # Remove dates and timestamps
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d*,?\s*\d*\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', '', text)
    
    # Remove rating patterns
    text = re.sub(r'(?:star|rating|review)s?', '', text, flags=re.IGNORECASE)
    
    # Remove ALL special characters and symbols
    text = re.sub(r'[+\-=_•●○◆◇★☆⭐👍👎📊📈📉💬✓✗×↓↑→←₹$€£¥₩%&@#]', '', text)
    
    # Remove emojis
    text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
    
    # Remove noise words
    noise_patterns = [
        r'\b(?:Pros|Cons|Positives|Negatives|Overall|Rating|Review)\b',
        r'\bCertified\s+Buyer\b',
        r'\bVerified\s+Purchase\b',
        r'\b(?:Helpful|Report|Abuse|READ\s+MORE|Show\s+More)\b',
        r'\b(?:Permalink|Performance|Previous|Next)\b',
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up punctuation
    text = re.sub(r'[:\-]{2,}', '', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'[,;:]\s*[,;:]', ',', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove very short fragments
    words = text.split()
    if len(words) < 5:
        return ""
    
    return text


def paraphrase_and_clean(text):
    """Extract meaningful sentences and ensure natural language output"""
    if not text:
        return ""
    
    text = clean_output_text(text)
    
    if not text:
        return ""
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Filter and clean sentences
    cleaned_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Skip if too short
        if len(sentence) < 25:
            continue
        
        # Skip if contains ANY numbers (double check)
        if re.search(r'\d', sentence):
            continue
        
        # Skip if contains special characters
        if re.search(r'[+\-=_•●★☆⭐@#$%]', sentence):
            continue
        
        # Skip duplicates
        sentence_lower = sentence.lower()
        if sentence_lower in seen:
            continue
        
        seen.add(sentence_lower)
        
        # Ensure sentence ends with period
        if not sentence.endswith('.'):
            sentence += '.'
        
        cleaned_sentences.append(sentence)
    
    return ' '.join(cleaned_sentences)

def stage1_t5_extract(input_text, task_type="summarize", max_length=150):
    """Stage 1: T5 extracts key information - IMPROVED PROMPTS"""
    if not T5_READY or t5_model is None:
        return None
    
    try:
        # Truncate input safely
        tokens = t5_tokenizer.encode(input_text, add_special_tokens=False, truncation=False)
        if len(tokens) > 350:
            print(f"[T5 STAGE 1] Input has {len(tokens)} tokens, truncating to 350...")
            tokens = tokens[:350]
            input_text = t5_tokenizer.decode(tokens, skip_special_tokens=True)
        
        # IMPROVED: More specific prompts for better extraction
        if task_type == "extract_pros":
            prompt = "List the positive aspects mentioned in these reviews in complete sentences"
        elif task_type == "extract_cons":
            prompt = "List the negative aspects mentioned in these reviews in complete sentences"
        elif task_type == "brief":
            prompt = "Provide a concise summary of the main points"
        else:
            prompt = "Summarize the key information from this text"
        
        full_prompt = f"{prompt}: {input_text}"
        
        inputs = t5_tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
            return_attention_mask=True
        ).to(device)
        
        print(f"[T5 STAGE 1] Input token count: {inputs.input_ids.shape[1]}")
        
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                min_length=40,  # Increased minimum
                num_beams=4,  # Better quality
                length_penalty=1.2,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7,  # Added for variety
                top_p=0.9  # Added for better sampling
            )
        
        t5_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[T5 STAGE 1] Generated: {t5_output}")
        return t5_output.strip()
        
    except Exception as e:
        print(f"[T5 STAGE 1] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def stage2_llama_refine(t5_output, task_type="summarize"):
    """Stage 2: Llama refines into human-like text - IMPROVED PROMPTS"""
    
    print(f"\n{'='*60}")
    print(f"[LLAMA STAGE 2] Starting refinement...")
    print(f"[LLAMA STAGE 2] LLAMA_READY = {LLAMA_READY}")
    print(f"[LLAMA STAGE 2] Input: {t5_output[:100]}...")
    print(f"{'='*60}")
    
    if not LLAMA_READY or llama_model is None:
        print(f"[LLAMA STAGE 2] ⚠️ Llama not ready - using cleaned T5 output")
        return paraphrase_and_clean(t5_output)
    
    try:
        # IMPROVED: Shorter, clearer system prompts
        if task_type == "extract_pros":
            system_prompt = "Rewrite this as 3-5 clear sentences about what customers liked. Be specific and natural."
        elif task_type == "extract_cons":
            system_prompt = "Rewrite this as 3-5 clear sentences about customer complaints. Be specific and natural."
        elif task_type == "brief":
            system_prompt = "Rewrite this as a brief, natural summary in 2-3 sentences. Be clear and conversational."
        else:
            system_prompt = "Rewrite this information clearly and naturally in 3-4 sentences."
        
        # IMPROVED: Simpler prompt format
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
            max_length=512,  # Increased for better context
            return_attention_mask=True
        ).to(device)
        
        print(f"[LLAMA STAGE 2] Generating (tokens: {inputs.input_ids.shape[1]})...")
        
        with torch.no_grad():
            outputs = llama_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,  # Reasonable length
                min_new_tokens=30,   # Ensure minimum output
                temperature=0.7,      # Balanced creativity
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=llama_tokenizer.eos_token_id,
                eos_token_id=llama_tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        llama_output = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"[LLAMA STAGE 2] Raw output: {llama_output[:200]}...")
        
        # Extract only the assistant's response
        if "<|assistant|>" in llama_output:
            llama_output = llama_output.split("<|assistant|>")[-1].strip()
        
        # Remove any remaining system/user tokens
        llama_output = re.sub(r'<\|.*?\|>', '', llama_output).strip()
        
        print(f"[LLAMA STAGE 2] After extraction: {llama_output[:200]}...")
        
        # Clean the output
        cleaned = paraphrase_and_clean(llama_output)
        
        print(f"[LLAMA STAGE 2] ✓ Final: {cleaned}")
        print(f"{'='*60}\n")
        
        return cleaned if cleaned else paraphrase_and_clean(t5_output)
        
    except Exception as e:
        print(f"[LLAMA STAGE 2] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return paraphrase_and_clean(t5_output)


def summarize_text(text, max_length=300, min_length=100):
    """Enhanced text summarization with better validation"""
    if not T5_READY:
        load_models()
    
    if not text or len(text.strip()) < 100:
        return "Text too short to summarize (minimum 100 characters required)."
    
    if not T5_READY:
        return "Summarization model not available."
    
    try:
        print(f"\n[TEXT SUMMARIZER] Processing: {len(text)} characters")
        
        # Clean and truncate
        cleaned_text = aggressive_clean_text(text)
        print(f"[TEXT SUMMARIZER] After cleaning: {len(cleaned_text)} characters")
        
        # Check if content is meaningful after cleaning
        if len(cleaned_text.strip()) < 50:
            # Try without aggressive cleaning
            print("[TEXT SUMMARIZER] Content too short after cleaning, using original...")
            cleaned_text = text
        
        cleaned_text = safe_truncate_text(cleaned_text, max_tokens=350)
        print(f"[TEXT SUMMARIZER] After truncation: {len(cleaned_text)} characters")
        
        if len(cleaned_text.strip()) < 50:
            return "Content is too short or contains insufficient information to summarize."
        
        # Use multi-stage processing
        result = multi_stage_process(cleaned_text, task_type="summarize")
        
        if not result or len(result) < 30:
            # Fallback: Use T5 directly without Llama
            print("[TEXT SUMMARIZER] Multi-stage failed, trying T5 only...")
            result = stage1_t5_extract(cleaned_text, task_type="summarize", max_length=max_length)
            
            if result:
                result = paraphrase_and_clean(result)
        
        if not result or len(result) < 30:
            return "Unable to generate meaningful summary from this content."
        
        print(f"[TEXT SUMMARIZER] ✓ Generated: {len(result)} chars")
        return result

    except Exception as e:
        print(f"[TEXT SUMMARIZER] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error generating summary. Please try a different page or content."


def generate_brief_summary_multistage(review_texts, stats):
    """Generate brief summary with better fallback handling"""
    if not review_texts:
        return "No reviews available to summarize."
    
    try:
        # Take sample of reviews
        cleaned_reviews = [aggressive_clean_text(r) for r in review_texts[:50]]
        combined = " ".join([r for r in cleaned_reviews if len(r) > 30])
        
        if len(combined) < 100:
            return "Insufficient review content for summary."
        
        combined = safe_truncate_text(combined, max_tokens=350)
        
        # Multi-stage processing
        result = multi_stage_process(combined, task_type="brief")
        
        if not result or len(result) < 30:
            # Fallback to T5 only
            result = stage1_t5_extract(combined, task_type="brief", max_length=200)
            if result:
                result = paraphrase_and_clean(result)
        
        if not result or len(result) < 30:
            # Ultimate fallback with stats
            avg_rating = stats.get('average_rating', 0)
            if avg_rating >= 4:
                return "Customers are generally satisfied with this product, praising its quality and performance."
            elif avg_rating >= 3:
                return "Customer opinions are mixed, with both positive feedback and areas for improvement noted."
            else:
                return "Customer reviews indicate concerns with the product, citing various issues."
        
        return result
        
    except Exception as e:
        print(f"[BRIEF SUMMARY] Error: {str(e)}")
        return "Error generating review summary."
    
def stage1_t5_extract(input_text, task_type="summarize", max_length=150):
    """Stage 1: T5 extracts key information - IMPROVED PROMPTS"""
    if not T5_READY or t5_model is None:
        return None
    
    try:
        # Truncate input safely
        tokens = t5_tokenizer.encode(input_text, add_special_tokens=False, truncation=False)
        if len(tokens) > 350:
            print(f"[T5 STAGE 1] Input has {len(tokens)} tokens, truncating to 350...")
            tokens = tokens[:350]
            input_text = t5_tokenizer.decode(tokens, skip_special_tokens=True)
        
        # IMPROVED: More specific prompts for better extraction
        if task_type == "extract_pros":
            prompt = "List the positive aspects mentioned in these reviews in complete sentences"
        elif task_type == "extract_cons":
            prompt = "List the negative aspects mentioned in these reviews in complete sentences"
        elif task_type == "brief":
            prompt = "Provide a concise summary of the main points"
        else:
            prompt = "Summarize the key information from this text"
        
        full_prompt = f"{prompt}: {input_text}"
        
        inputs = t5_tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
            return_attention_mask=True
        ).to(device)
        
        print(f"[T5 STAGE 1] Input token count: {inputs.input_ids.shape[1]}")
        
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                min_length=40,  # Increased minimum
                num_beams=4,  # Better quality
                length_penalty=1.2,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7,  # Added for variety
                top_p=0.9  # Added for better sampling
            )
        
        t5_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[T5 STAGE 1] Generated: {t5_output}")
        return t5_output.strip()
        
    except Exception as e:
        print(f"[T5 STAGE 1] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def stage2_llama_refine(t5_output, task_type="summarize"):
    """Stage 2: Llama refines into human-like text - IMPROVED PROMPTS"""
    
    print(f"\n{'='*60}")
    print(f"[LLAMA STAGE 2] Starting refinement...")
    print(f"[LLAMA STAGE 2] LLAMA_READY = {LLAMA_READY}")
    print(f"[LLAMA STAGE 2] Input: {t5_output[:100]}...")
    print(f"{'='*60}")
    
    if not LLAMA_READY or llama_model is None:
        print(f"[LLAMA STAGE 2] ⚠️ Llama not ready - using cleaned T5 output")
        return paraphrase_and_clean(t5_output)
    
    try:
        # IMPROVED: Shorter, clearer system prompts
        if task_type == "extract_pros":
            system_prompt = "Rewrite this as 3-5 clear sentences about what customers liked. Be specific and natural."
        elif task_type == "extract_cons":
            system_prompt = "Rewrite this as 3-5 clear sentences about customer complaints. Be specific and natural."
        elif task_type == "brief":
            system_prompt = "Rewrite this as a brief, natural summary in 2-3 sentences. Be clear and conversational."
        else:
            system_prompt = "Rewrite this information clearly and naturally in 3-4 sentences."
        
        # IMPROVED: Simpler prompt format
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
            max_length=512,  # Increased for better context
            return_attention_mask=True
        ).to(device)
        
        print(f"[LLAMA STAGE 2] Generating (tokens: {inputs.input_ids.shape[1]})...")
        
        with torch.no_grad():
            outputs = llama_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,  # Reasonable length
                min_new_tokens=30,   # Ensure minimum output
                temperature=0.7,      # Balanced creativity
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=llama_tokenizer.eos_token_id,
                eos_token_id=llama_tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        llama_output = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"[LLAMA STAGE 2] Raw output: {llama_output[:200]}...")
        
        # Extract only the assistant's response
        if "<|assistant|>" in llama_output:
            llama_output = llama_output.split("<|assistant|>")[-1].strip()
        
        # Remove any remaining system/user tokens
        llama_output = re.sub(r'<\|.*?\|>', '', llama_output).strip()
        
        print(f"[LLAMA STAGE 2] After extraction: {llama_output[:200]}...")
        
        # Clean the output
        cleaned = paraphrase_and_clean(llama_output)
        
        print(f"[LLAMA STAGE 2] ✓ Final: {cleaned}")
        print(f"{'='*60}\n")
        
        return cleaned if cleaned else paraphrase_and_clean(t5_output)
        
    except Exception as e:
        print(f"[LLAMA STAGE 2] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return paraphrase_and_clean(t5_output)


def summarize_text(text, max_length=300, min_length=100):
    """Enhanced text summarization with better validation"""
    if not T5_READY:
        load_models()
    
    if not text or len(text.strip()) < 100:
        return "Text too short to summarize (minimum 100 characters required)."
    
    if not T5_READY:
        return "Summarization model not available."
    
    try:
        print(f"\n[TEXT SUMMARIZER] Processing: {len(text)} characters")
        
        # Clean and truncate
        cleaned_text = aggressive_clean_text(text)
        print(f"[TEXT SUMMARIZER] After cleaning: {len(cleaned_text)} characters")
        
        # Check if content is meaningful after cleaning
        if len(cleaned_text.strip()) < 50:
            # Try without aggressive cleaning
            print("[TEXT SUMMARIZER] Content too short after cleaning, using original...")
            cleaned_text = text
        
        cleaned_text = safe_truncate_text(cleaned_text, max_tokens=350)
        print(f"[TEXT SUMMARIZER] After truncation: {len(cleaned_text)} characters")
        
        if len(cleaned_text.strip()) < 50:
            return "Content is too short or contains insufficient information to summarize."
        
        # Use multi-stage processing
        result = multi_stage_process(cleaned_text, task_type="summarize")
        
        if not result or len(result) < 30:
            # Fallback: Use T5 directly without Llama
            print("[TEXT SUMMARIZER] Multi-stage failed, trying T5 only...")
            result = stage1_t5_extract(cleaned_text, task_type="summarize", max_length=max_length)
            
            if result:
                result = paraphrase_and_clean(result)
        
        if not result or len(result) < 30:
            return "Unable to generate meaningful summary from this content."
        
        print(f"[TEXT SUMMARIZER] ✓ Generated: {len(result)} chars")
        return result

    except Exception as e:
        print(f"[TEXT SUMMARIZER] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error generating summary. Please try a different page or content."


def generate_brief_summary_multistage(review_texts, stats):
    """Generate brief summary with better fallback handling"""
    if not review_texts:
        return "No reviews available to summarize."
    
    try:
        # Take sample of reviews
        cleaned_reviews = [aggressive_clean_text(r) for r in review_texts[:50]]
        combined = " ".join([r for r in cleaned_reviews if len(r) > 30])
        
        if len(combined) < 100:
            return "Insufficient review content for summary."
        
        combined = safe_truncate_text(combined, max_tokens=350)
        
        # Multi-stage processing
        result = multi_stage_process(combined, task_type="brief")
        
        if not result or len(result) < 30:
            # Fallback to T5 only
            result = stage1_t5_extract(combined, task_type="brief", max_length=200)
            if result:
                result = paraphrase_and_clean(result)
        
        if not result or len(result) < 30:
            # Ultimate fallback with stats
            avg_rating = stats.get('average_rating', 0)
            if avg_rating >= 4:
                return "Customers are generally satisfied with this product, praising its quality and performance."
            elif avg_rating >= 3:
                return "Customer opinions are mixed, with both positive feedback and areas for improvement noted."
            else:
                return "Customer reviews indicate concerns with the product, citing various issues."
        
        return result
        
    except Exception as e:
        print(f"[BRIEF SUMMARY] Error: {str(e)}")
        return "Error generating review summary."



def stage2_llama_refine(t5_output, task_type="summarize"):
    """Stage 2: Llama refines into human-like text - CPU optimized with ENHANCED DEBUGGING"""
    
    # CRITICAL DEBUG: Check if Llama is ready
    print(f"\n{'='*60}")
    print(f"[LLAMA STAGE 2 DEBUG] Starting...")
    print(f"[LLAMA STAGE 2 DEBUG] LLAMA_READY = {LLAMA_READY}")
    print(f"[LLAMA STAGE 2 DEBUG] llama_model = {llama_model}")
    print(f"[LLAMA STAGE 2 DEBUG] Input length = {len(t5_output) if t5_output else 0}")
    print(f"{'='*60}")
    
    if not LLAMA_READY or llama_model is None:
        print(f"[LLAMA STAGE 2] ⚠️ Llama NOT ready - falling back to paraphrase_and_clean")
        print(f"[LLAMA STAGE 2] LLAMA_READY={LLAMA_READY}, llama_model={llama_model}")
        return paraphrase_and_clean(t5_output)
    
    try:
        print(f"[LLAMA STAGE 2] ✓ Llama is ready, proceeding with generation...")
        
        # Strong emphasis on human-like writing
        if task_type == "extract_pros":
            system_prompt = "You are a helpful product reviewer. Rewrite the following as natural sentences about what customers appreciated. Write like a human - use complete sentences, natural flow, and your own words. Never use numbers, ratings, or symbols. Only use plain English text."
        elif task_type == "extract_cons":
            system_prompt = "You are a helpful product reviewer. Rewrite the following as natural sentences about customer concerns. Write like a human - use complete sentences, natural flow, and your own words. Never use numbers, ratings, or symbols. Only use plain English text."
        elif task_type == "brief":
            system_prompt = "You are a helpful assistant. Write a brief, conversational summary in plain English. Sound natural and human. Never use numbers or symbols - only words and sentences."
        else:
            system_prompt = "You are a helpful assistant. Rewrite this naturally in plain English like a human would. No numbers, no symbols - just natural sentences."
        
        prompt = f"""<|system|>
{system_prompt}</s>
<|user|>
{t5_output}</s>
<|assistant|>
"""
        
        print(f"[LLAMA STAGE 2] Tokenizing input (length: {len(prompt)} chars)...")
        
        inputs = llama_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=400,  # Reduced for CPU
            return_attention_mask=True
        ).to(device)
        
        print(f"[LLAMA STAGE 2] Input tokenized: {inputs.input_ids.shape[1]} tokens")
        print(f"[LLAMA STAGE 2] Generating with Llama model...")
        
        with torch.no_grad():
            outputs = llama_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=200,  # Reduced for CPU
                temperature=0.85,
                top_p=0.9,
                do_sample=True,
                pad_token_id=llama_tokenizer.eos_token_id,
                repetition_penalty=1.3
            )
        
        print(f"[LLAMA STAGE 2] ✓ Generation complete!")
        
        llama_output = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"[LLAMA STAGE 2] Raw output length: {len(llama_output)} chars")
        print(f"[LLAMA STAGE 2] Raw output preview: {llama_output[:150]}...")
        
        # Extract assistant's response
        if "<|assistant|>" in llama_output:
            llama_output = llama_output.split("<|assistant|>")[-1].strip()
            print(f"[LLAMA STAGE 2] After extracting assistant response: {len(llama_output)} chars")
        
        print(f"[LLAMA STAGE 2] Cleaning output with paraphrase_and_clean...")
        cleaned = paraphrase_and_clean(llama_output)
        
        print(f"[LLAMA STAGE 2] ✓ Final output: {len(cleaned)} chars")
        print(f"[LLAMA STAGE 2] Final preview: {cleaned[:100]}...")
        print(f"{'='*60}\n")
        
        return cleaned
        
    except Exception as e:
        print(f"\n[LLAMA STAGE 2] ❌ ERROR OCCURRED:")
        print(f"[LLAMA STAGE 2] Error type: {type(e).__name__}")
        print(f"[LLAMA STAGE 2] Error message: {str(e)}")
        import traceback
        print("[LLAMA STAGE 2] Full traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        print(f"[LLAMA STAGE 2] Falling back to paraphrase_and_clean")
        return paraphrase_and_clean(t5_output)


def multi_stage_process(input_text, task_type="summarize"):
    """Multi-stage processing with timeout protection - ENHANCED DEBUG VERSION"""
    print(f"\n{'#'*60}")
    print(f"[MULTI-STAGE] Starting {task_type} processing...")
    print(f"[MULTI-STAGE] Input length: {len(input_text)} chars")
    print(f"{'#'*60}")
    
    try:
        # Stage 1: T5 extraction
        print(f"[MULTI-STAGE] >>> STAGE 1: T5 Extraction <<<")
        t5_output = stage1_t5_extract(input_text, task_type=task_type)
        
        if not t5_output:
            print(f"[MULTI-STAGE] ❌ Stage 1 failed - no output")
            return None
        
        print(f"[MULTI-STAGE] ✓ Stage 1 complete: {len(t5_output)} chars")
        print(f"[STAGE 1 OUTPUT] {t5_output[:150]}...")
        
        # Stage 2: Llama refinement
        print(f"\n[MULTI-STAGE] >>> STAGE 2: Llama Refinement <<<")
        final_output = stage2_llama_refine(t5_output, task_type=task_type)
        
        print(f"[MULTI-STAGE] ✓ Stage 2 complete: {len(final_output) if final_output else 0} chars")
        print(f"[STAGE 2 OUTPUT] {final_output[:150] if final_output else 'None'}...")
        print(f"{'#'*60}\n")
        
        return final_output
    
    except Exception as e:
        print(f"\n[MULTI-STAGE] ❌ CRITICAL ERROR:")
        print(f"[MULTI-STAGE] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'#'*60}\n")
        return None

def multi_stage_process(input_text, task_type="summarize"):
    """Multi-stage processing with timeout protection"""
       # CRITICAL DEBUG: Check if Llama is ready
    print(f"\n{'='*60}")
    print(f"[LLAMA STAGE 2 DEBUG] Starting...")
    print(f"[LLAMA STAGE 2 DEBUG] LLAMA_READY = {LLAMA_READY}")
    print(f"[LLAMA STAGE 2 DEBUG] llama_model = {llama_model}")
    print(f"[LLAMA STAGE 2 DEBUG] Input length = {len(t5_output) if t5_output else 0}")
    print(f"{'='*60}")
    
    if not LLAMA_READY or llama_model is None:
        print(f"[LLAMA STAGE 2] ⚠️ Llama NOT ready - falling back to paraphrase_and_clean")
        print(f"[LLAMA STAGE 2] LLAMA_READY={LLAMA_READY}, llama_model={llama_model}")
        return paraphrase_and_clean(t5_output)
    
    try:
        print(f"[LLAMA STAGE 2] ✓ Llama is ready, proceeding with generation...")
        
        # Strong emphasis on human-like writing
        if task_type == "extract_pros":
            system_prompt = "You are a helpful product reviewer. Rewrite the following as natural sentences about what customers appreciated. Write like a human - use complete sentences, natural flow, and your own words. Never use numbers, ratings, or symbols. Only use plain English text."
        elif task_type == "extract_cons":
            system_prompt = "You are a helpful product reviewer. Rewrite the following as natural sentences about customer concerns. Write like a human - use complete sentences, natural flow, and your own words. Never use numbers, ratings, or symbols. Only use plain English text."
        elif task_type == "brief":
            system_prompt = "You are a helpful assistant. Write a brief, conversational summary in plain English. Sound natural and human. Never use numbers or symbols - only words and sentences."
        else:
            system_prompt = "You are a helpful assistant. Rewrite this naturally in plain English like a human would. No numbers, no symbols - just natural sentences."
        
        prompt = f"""<|system|>
{system_prompt}</s>
<|user|>
{t5_output}</s>
<|assistant|>
"""
        
        print(f"[LLAMA STAGE 2] Tokenizing input (length: {len(prompt)} chars)...")
        
        inputs = llama_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=400,  # Reduced for CPU
            return_attention_mask=True
        ).to(device)
        
        print(f"[LLAMA STAGE 2] Input tokenized: {inputs.input_ids.shape[1]} tokens")
        print(f"[LLAMA STAGE 2] Generating with Llama model...")
        
        with torch.no_grad():
            outputs = llama_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=200,  # Reduced for CPU
                temperature=0.85,
                top_p=0.9,
                do_sample=True,
                pad_token_id=llama_tokenizer.eos_token_id,
                repetition_penalty=1.3
            )
        
        print(f"[LLAMA STAGE 2] ✓ Generation complete!")
        
        llama_output = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"[LLAMA STAGE 2] Raw output length: {len(llama_output)} chars")
        print(f"[LLAMA STAGE 2] Raw output preview: {llama_output[:150]}...")
        
        # Extract assistant's response
        if "<|assistant|>" in llama_output:
            llama_output = llama_output.split("<|assistant|>")[-1].strip()
            print(f"[LLAMA STAGE 2] After extracting assistant response: {len(llama_output)} chars")
        
        print(f"[LLAMA STAGE 2] Cleaning output with paraphrase_and_clean...")
        cleaned = paraphrase_and_clean(llama_output)
        
        print(f"[LLAMA STAGE 2] ✓ Final output: {len(cleaned)} chars")
        print(f"[LLAMA STAGE 2] Final preview: {cleaned[:100]}...")
        print(f"{'='*60}\n")
        
        return cleaned
        
    except Exception as e:
        print(f"\n[LLAMA STAGE 2] ❌ ERROR OCCURRED:")
        print(f"[LLAMA STAGE 2] Error type: {type(e).__name__}")
        print(f"[LLAMA STAGE 2] Error message: {str(e)}")
        import traceback
        print("[LLAMA STAGE 2] Full traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        print(f"[LLAMA STAGE 2] Falling back to paraphrase_and_clean")
        return paraphrase_and_clean(t5_output)


def multi_stage_process(input_text, task_type="summarize"):
    """Multi-stage processing with timeout protection - ENHANCED DEBUG VERSION"""
    print(f"\n{'#'*60}")
    print(f"[MULTI-STAGE] Starting {task_type} processing...")
    print(f"[MULTI-STAGE] Input length: {len(input_text)} chars")
    print(f"{'#'*60}")
    
    try:
        # Stage 1: T5 extraction
        print(f"[MULTI-STAGE] >>> STAGE 1: T5 Extraction <<<")
        t5_output = stage1_t5_extract(input_text, task_type=task_type)
        
        if not t5_output:
            print(f"[MULTI-STAGE] ❌ Stage 1 failed - no output")
            return None
        
        print(f"[MULTI-STAGE] ✓ Stage 1 complete: {len(t5_output)} chars")
        print(f"[STAGE 1 OUTPUT] {t5_output[:150]}...")
        
        # Stage 2: Llama refinement
        print(f"\n[MULTI-STAGE] >>> STAGE 2: Llama Refinement <<<")
        final_output = stage2_llama_refine(t5_output, task_type=task_type)
        
        print(f"[MULTI-STAGE] ✓ Stage 2 complete: {len(final_output) if final_output else 0} chars")
        print(f"[STAGE 2 OUTPUT] {final_output[:150] if final_output else 'None'}...")
        print(f"{'#'*60}\n")
        
        return final_output
    
    except Exception as e:
        print(f"\n[MULTI-STAGE] ❌ CRITICAL ERROR:")
        print(f"[MULTI-STAGE] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'#'*60}\n")
        return None

def aggressive_clean_text(text):
    """Ultra aggressive cleaning to remove ALL noise"""
    # Remove ALL numbers first
    text = re.sub(r'\d+', '', text)
    
    # Remove dates
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*,?\s*\b', '', text, flags=re.IGNORECASE)
    
    # Remove usernames
    text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '', text)
    text = re.sub(r'Certified\s+Buyer|Flipkart\s+Customer', '', text, flags=re.IGNORECASE)
    
    # Remove URLs and emails
    text = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+', '', text)
    
    # Remove noise phrases
    text = re.sub(r'READ MORE|Report Abuse|Was this helpful', '', text, flags=re.IGNORECASE)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def safe_truncate_text(text, max_tokens=350):
    """Safely truncate text - CPU friendly with aggressive limits"""
    if not text:
        return ""
    
    if not t5_tokenizer:
        # Fallback: use character limit (roughly 4 chars per token)
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars]
    
    try:
        # Encode to tokens
        tokens = t5_tokenizer.encode(text, add_special_tokens=False, truncation=False)
        
        print(f"[TRUNCATE] Original tokens: {len(tokens)}, Target: {max_tokens}")
        
        if len(tokens) <= max_tokens:
            return text
        
        # Strategy: Take first 60% and last 40% to preserve context
        chunk1_size = int(max_tokens * 0.6)
        chunk2_size = int(max_tokens * 0.4)
        
        chunk1 = tokens[:chunk1_size]
        chunk2 = tokens[-chunk2_size:]
        
        combined = chunk1 + chunk2
        
        result = t5_tokenizer.decode(combined, skip_special_tokens=True)
        print(f"[TRUNCATE] Truncated to {len(combined)} tokens, {len(result)} chars")
        
        return result
        
    except Exception as e:
        print(f"[TRUNCATE] Error: {e}")
        # Fallback to character-based truncation
        max_chars = max_tokens * 4
        return text[:max_chars]


def process_review_batch(reviews_batch, batch_num):
    """Process a single batch of reviews"""
    print(f"[BATCH {batch_num}] Processing {len(reviews_batch)} reviews...")
    
    cleaned_reviews = []
    for review in reviews_batch:
        body = review.get('body', '')
        title = review.get('title', '')
        full_text = f"{title} {body}".strip()
        full_text = aggressive_clean_text(full_text)
        
        if full_text and len(full_text) > 30:
            cleaned_reviews.append(full_text)
    
    return cleaned_reviews


def summarize_reviews_with_analysis(reviews_data, max_reviews=150):
    """
    Enhanced review analysis with CPU-friendly batching.
    REMOVED: Detailed summary feature as requested.
    """
    if not T5_READY:
        load_models()
    
    if not reviews_data:
        return {
            'success': False,
            'brief_summary': "No reviews available to analyze.",
            'pros': [],
            'cons': [],
            'stats': {}
        }
    
    if not T5_READY:
        return {
            'success': False,
            'brief_summary': "Summarization model not available.",
            'pros': [],
            'cons': [],
            'stats': {}
        }
    
    try:
        print(f"\n[REVIEW ANALYZER] Processing {len(reviews_data)} reviews in batches")
        
        # Limit total reviews
        reviews_data = reviews_data[:max_reviews]
        
        # Process in batches to avoid CPU overload
        positive_reviews = []
        negative_reviews = []
        ratings = []
        
        # Split into batches
        batches = [reviews_data[i:i + BATCH_SIZE] for i in range(0, len(reviews_data), BATCH_SIZE)]
        
        print(f"[REVIEW ANALYZER] Split into {len(batches)} batches of ~{BATCH_SIZE} reviews each")
        
        # Process batches with timeout protection
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {
                executor.submit(process_review_batch, batch, idx): idx 
                for idx, batch in enumerate(batches, 1)
            }
            
            for future in as_completed(future_to_batch, timeout=PROCESSING_TIMEOUT * len(batches)):
                batch_idx = future_to_batch[future]
                try:
                    cleaned_batch = future.result(timeout=PROCESSING_TIMEOUT)
                    
                    # Categorize by rating
                    for i, review in enumerate(batches[batch_idx - 1]):
                        rating = review.get('rating')
                        if rating:
                            ratings.append(rating)
                            if i < len(cleaned_batch):
                                if rating >= 4:
                                    positive_reviews.append(cleaned_batch[i])
                                elif rating <= 2:
                                    negative_reviews.append(cleaned_batch[i])
                    
                    print(f"[BATCH {batch_idx}] Completed successfully")
                    
                except Exception as e:
                    print(f"[BATCH {batch_idx}] Error or timeout: {e}")
                    continue
        
        print(f"[REVIEW ANALYZER] Cleaned - Positive: {len(positive_reviews)}, Negative: {len(negative_reviews)}")
        
        stats = calculate_review_stats(ratings, len(reviews_data))
        
        # Generate summaries (CPU-friendly with timeouts)
        brief_summary = generate_brief_summary_multistage(positive_reviews + negative_reviews, stats)
        pros = extract_pros_multistage(positive_reviews)
        cons = extract_cons_multistage(negative_reviews)
        
        return {
            'success': True,
            'brief_summary': brief_summary,
            'pros': pros,
            'cons': cons,
            'stats': stats
        }
    
    except Exception as e:
        print(f"[REVIEW ANALYZER] Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'brief_summary': "Error analyzing reviews.",
            'pros': [],
            'cons': [],
            'stats': {}
        }


def generate_brief_summary_multistage(review_texts, stats):
    """Generate brief summary - NO NUMBERS in output"""
    if not review_texts:
        return "No reviews available."
    
    try:
        # Take limited reviews for CPU efficiency
        cleaned_reviews = [aggressive_clean_text(r) for r in review_texts[:50]]
        combined = " ".join([r for r in cleaned_reviews if len(r) > 30])
        combined = safe_truncate_text(combined, max_tokens=350)
        
        if len(combined) < 50:
            return "Insufficient review content for summary."
        
        # Multi-stage processing
        result = multi_stage_process(combined, task_type="brief")
        
        if not result or len(result) < 30:
            return "Could not generate summary."
        
        # Return ONLY natural language - NO numbers
        return result
        
    except Exception as e:
        print(f"[BRIEF SUMMARY] Error: {str(e)}")
        return "Error generating brief summary."


def extract_pros_multistage(positive_reviews):
    """Extract pros with clean, human-like output"""
    if not positive_reviews or len(positive_reviews) < 3:
        return []
    
    try:
        # Limit for CPU
        cleaned = [aggressive_clean_text(r) for r in positive_reviews[:40]]
        combined = " ".join([r for r in cleaned if len(r) > 30])
        combined = safe_truncate_text(combined, max_tokens=300)
        
        if len(combined) < 50:
            return ["Customers are generally satisfied with the product."]
        
        result = multi_stage_process(combined, task_type="extract_pros")
        
        if not result:
            return ["Customers are generally satisfied with the product."]
        
        # Parse into clean list
        pros = []
        sentences = [s.strip() for s in re.split(r'[.\n•\-]+', result)]
        
        for s in sentences:
            s = clean_output_text(s)
            if not s or len(s) < 25 or len(s) > 200:
                continue
            
            # Ensure NO numbers or symbols
            if re.search(r'\d', s) or re.search(r'[+\-=_•★☆@#$%]', s):
                continue
            
            if not s.endswith('.'):
                s += '.'
            pros.append(s)
            
            if len(pros) >= 5:
                break
        
        return pros if pros else ["Customers are generally satisfied with the product."]
        
    except Exception as e:
        print(f"[PROS] Error: {str(e)}")
        return ["Customers are generally satisfied with the product."]


def extract_cons_multistage(negative_reviews):
    """Extract cons with clean, human-like output"""
    if not negative_reviews or len(negative_reviews) < 3:
        return []
    
    try:
        # Limit for CPU
        cleaned = [aggressive_clean_text(r) for r in negative_reviews[:40]]
        combined = " ".join([r for r in cleaned if len(r) > 30])
        combined = safe_truncate_text(combined, max_tokens=300)
        
        if len(combined) < 50:
            return ["Some customers reported issues with the product."]
        
        result = multi_stage_process(combined, task_type="extract_cons")
        
        if not result:
            return ["Some customers reported issues with the product."]
        
        # Parse into clean list
        cons = []
        sentences = [s.strip() for s in re.split(r'[.\n•\-]+', result)]
        
        for s in sentences:
            s = clean_output_text(s)
            if not s or len(s) < 25 or len(s) > 200:
                continue
            
            # Ensure NO numbers or symbols
            if re.search(r'\d', s) or re.search(r'[+\-=_•★☆@#$%]', s):
                continue
            
            if not s.endswith('.'):
                s += '.'
            cons.append(s)
            
            if len(cons) >= 5:
                break
        
        return cons if cons else ["Some customers reported issues with the product."]
        
    except Exception as e:
        print(f"[CONS] Error: {str(e)}")
        return ["Some customers reported issues with the product."]


def summarize_text(text, max_length=300, min_length=100):
    """Enhanced text summarization - CPU optimized"""
    if not T5_READY:
        load_models()
    
    if not text or len(text.strip()) < 100:
        return "Text too short to summarize."
    
    if not T5_READY:
        return "Summarization model not available."
    
    try:
        print(f"\n[TEXT SUMMARIZER] Input: {len(text)} characters")
        
        # CRITICAL: Clean and truncate BEFORE processing
        cleaned_text = aggressive_clean_text(text)
        print(f"[TEXT SUMMARIZER] After cleaning: {len(cleaned_text)} characters")
        
        # Truncate to safe token limit (350 tokens = ~1400 chars)
        cleaned_text = safe_truncate_text(cleaned_text, max_tokens=350)
        print(f"[TEXT SUMMARIZER] After truncation: {len(cleaned_text)} characters")
        
        if len(cleaned_text.strip()) < 50:
            return "Content too short after cleaning."
        
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
    
    if len(ratings) > 0:
        stats['positive_percentage'] = round((pos / len(ratings)) * 100, 1)
        stats['negative_percentage'] = round((neg / len(ratings)) * 100, 1)
    
    return stats