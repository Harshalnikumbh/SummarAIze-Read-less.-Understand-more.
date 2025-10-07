import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow warnings
from transformers import pipeline
import torch
import re

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Model state
MODEL_READY = False
summarizer = None

def load_model():
    """Load the summarization model (call this from Flask app)"""
    global MODEL_READY, summarizer
    
    if MODEL_READY:
        return  # Already loaded
    
    print("\n" + "="*60)
    print("📦 Loading summarization model...")
    print("="*60)
    
    try:
        # BART large model with better configuration
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn", 
            device=device,
            model_kwargs={"max_length": 1024}
        )
        MODEL_READY = True
        print("✓ BART model loaded successfully")
        print("="*60 + "\n")
    except Exception as e:
        print(f"⚠️  Error loading BART: {e}")
        # Fallback to smaller model
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
            MODEL_READY = True
            print("✓ DistilBART model loaded successfully (fallback)")
            print("="*60 + "\n")
        except Exception as e2:
            print(f"❌ Error loading models: {e2}")
            summarizer = None
            print("="*60 + "\n")

# Don't load model here - let Flask app load it when ready


def summarize_text(text, max_length=350, min_length=150):
    """
    Summarize any text content using BART with enhanced detail.
    
    Args:
        text: Text content to summarize
        max_length: Maximum summary length (increased default)
        min_length: Minimum summary length (increased default)
        
    Returns:
        Summary string
    """
    # Auto-load model if not loaded
    if not MODEL_READY:
        load_model()
    
    if not text or len(text.strip()) < 100:
        return "Text too short to summarize."
    
    if not MODEL_READY or summarizer is None:
        return "Summarization model not available."
    
    try:
        print(f"\n[SUMMARIZER] Input text length: {len(text)} characters")
        
        # Clean the text
        cleaned_text = clean_text(text)
        print(f"[SUMMARIZER] Cleaned text length: {len(cleaned_text)} characters")
        
        # CRITICAL: BART has strict 1024 token limit (~700-750 words safe)
        # Using conservative limit to avoid IndexError
        max_input_words = 650  # Safe limit to avoid token overflow
        words = cleaned_text.split()
        print(f"[SUMMARIZER] Word count: {len(words)}")
        
        # Better handling of long text - preserve more context
        if len(words) > max_input_words:
            print(f"[SUMMARIZER] Truncating from {len(words)} to {max_input_words} words")
            
            # Take more from beginning and end to preserve context
            first_size = int(max_input_words * 0.6)  # 60% from start
            last_size = int(max_input_words * 0.4)   # 40% from end
            
            first_size = min(first_size, len(words))
            last_size = min(last_size, len(words))
            
            first_part = words[:first_size]
            last_part = words[-last_size:] if last_size > 0 else []
            
            words = first_part + last_part
            cleaned_text = " ".join(words)
            print(f"[SUMMARIZER] After truncation: {len(words)} words")
        
        # Double-check we're still within limits
        if len(cleaned_text.split()) > max_input_words:
            cleaned_text = " ".join(cleaned_text.split()[:max_input_words])
            print(f"[SUMMARIZER] Safety truncation applied")
        
        # Make sure we still have enough content
        if len(cleaned_text.split()) < 50:
            print("[SUMMARIZER] Content too short after cleaning")
            return extractive_summary(text)
        
        # IMPROVED: Dynamic length calculation based on input
        # For longer content, generate longer summaries
        word_count = len(words)
        
        # Scale summary length with input length - MORE AGGRESSIVE
        if word_count > 600:
            adjusted_max_length = 500  # Much longer for long content
            adjusted_min_length = 250
        elif word_count > 400:
            adjusted_max_length = 400
            adjusted_min_length = 200
        elif word_count > 250:
            adjusted_max_length = 300
            adjusted_min_length = 150
        else:
            adjusted_max_length = 250
            adjusted_min_length = 100
        
        # Ensure max is always greater than min
        adjusted_max_length = max(adjusted_max_length, adjusted_min_length + 50)
        
        print(f"[SUMMARIZER] Generating summary (max={adjusted_max_length}, min={adjusted_min_length})")
        
        # ENHANCED: Generate summary with better parameters
        # CRITICAL: Always enforce truncation to prevent IndexError
        summary = summarizer(
            cleaned_text,
            max_length=adjusted_max_length,
            min_length=adjusted_min_length,
            do_sample=False,
            truncation=True,  # MUST be True
            max_new_tokens=adjusted_max_length,  # Explicit token limit
            # These parameters encourage more detailed summaries
            num_beams=4,  # Use beam search for better quality
            length_penalty=1.0,  # Neutral penalty (don't favor short summaries)
            early_stopping=True,
            no_repeat_ngram_size=3  # Avoid repetition
        )
        
        result = summary[0]['summary_text']
        print(f"[SUMMARIZER] Summary generated: {len(result)} characters, {len(result.split())} words")
        
        # Post-process: Ensure proper formatting
        result = post_process_summary(result)
        
        return result
    
    except Exception as e:
        print(f"[SUMMARIZER] Error during summarization: {str(e)}")
        import traceback
        traceback.print_exc()
        return extractive_summary(text)


def summarize_reviews(review_texts, max_reviews=50):
    """
    Summarize a list of reviews using BART with enhanced detail.
    
    Args:
        review_texts: List of review strings
        max_reviews: Maximum number of reviews to process
        
    Returns:
        Summary string
    """
    # Auto-load model if not loaded
    if not MODEL_READY:
        load_model()
    
    if not review_texts:
        return "No reviews available to summarize."
    
    if not MODEL_READY or summarizer is None:
        return "Summarization model not available."
    
    try:
        # Limit number of reviews
        review_texts = review_texts[:max_reviews]
        
        # Combine and clean text
        combined_text = " ".join([clean_text(r) for r in review_texts])
        
        # Handle token limits - CRITICAL: BART max is 1024 tokens
        max_input_words = 600  # Safe limit for reviews
        words = combined_text.split()
        
        if len(words) > max_input_words:
            # Sample from beginning, middle, and end
            first_size = int(max_input_words * 0.35)
            middle_size = int(max_input_words * 0.30)
            last_size = int(max_input_words * 0.35)
            
            first_size = min(first_size, len(words))
            last_size = min(last_size, len(words))
            
            middle_start = max(0, (len(words) // 2) - (middle_size // 2))
            middle_end = min(len(words), middle_start + middle_size)
            
            first_part = words[:first_size]
            middle_part = words[middle_start:middle_end]
            last_part = words[-last_size:] if last_size > 0 else []
            
            words = first_part + middle_part + last_part
            combined_text = " ".join(words)
        
        # IMPROVED: Better parameters for review summarization
        # CRITICAL: Always enforce truncation
        summary = summarizer(
            combined_text,
            max_length=250,  # Increased from 150
            min_length=100,  # Increased from 40
            do_sample=False,
            truncation=True, 
            max_new_tokens=250,  # Explicit token limit
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        result = summary[0]['summary_text']
        return post_process_summary(result)
    
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return extractive_summary(review_texts)

def clean_text(text):
    """Clean text for summarization while preserving important content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'\""]', '', text)
    
    # Keep words that are 2+ characters (removed the filtering that was too aggressive)
    text = ' '.join([w for w in text.split() if len(w) > 1 or w in '.,!?'])
    
    return text.strip()


def post_process_summary(summary):
    """
    Post-process the summary to ensure proper formatting and readability.
    """
    # Ensure proper sentence endings
    if not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    # Fix spacing after punctuation
    summary = re.sub(r'([.!?])([A-Z])', r'\1 \2', summary)
    
    # Remove extra spaces
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    # Capitalize first letter
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    return summary


def extractive_summary(text, num_sentences=8):
    """
    Enhanced fallback extractive summary.
    Takes more representative sentences for better detail.
    """
    if isinstance(text, list):
        # Handle list of reviews
        all_sentences = []
        for review in text[:15]:  # Increased from 10
            sentences = re.split(r'[.!?]+', review)
            all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 20])
    else:
        # Handle single text
        sentences = re.split(r'[.!?]+', text)
        all_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if all_sentences:
        # Take more sentences for better coverage
        # Use beginning, middle, and end
        num_sentences = min(num_sentences, len(all_sentences))
        
        if len(all_sentences) <= num_sentences:
            summary_sentences = all_sentences
        else:
            # Intelligent sampling
            start_idx = min(3, len(all_sentences))
            middle_idx = len(all_sentences) // 2
            end_start = max(len(all_sentences) - 3, start_idx + 1)
            
            summary_sentences = (
                all_sentences[:start_idx] +
                [all_sentences[middle_idx]] +
                all_sentences[end_start:]
            )
            summary_sentences = summary_sentences[:num_sentences]
        
        return ". ".join(summary_sentences) + "."
    
    return "Content available but summary could not be generated."