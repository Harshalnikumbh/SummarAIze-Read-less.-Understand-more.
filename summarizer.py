import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import pipeline, AutoTokenizer
import torch
import re
from collections import Counter

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Model state
MODEL_READY = False
summarizer = None
tokenizer = None

def load_model():
    """Load the summarization model with tokenizer"""
    global MODEL_READY, summarizer, tokenizer
    
    if MODEL_READY:
        return
    
    print("\n" + "="*60)
    print("📦 Loading BART summarization model...")
    print("="*60)
    
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        summarizer = pipeline(
            "summarization", 
            model=model_name,
            tokenizer=tokenizer,
            device=device,
            framework="pt"
        )
        MODEL_READY = True
        print("✓ BART model loaded successfully")
        print(f"✓ Max input tokens: 1024")
        print("="*60 + "\n")
    except Exception as e:
        print(f"⚠️  Error loading BART: {e}")
        try:
            model_name = "sshleifer/distilbart-cnn-12-6"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            summarizer = pipeline("summarization", model=model_name, tokenizer=tokenizer, device=device)
            MODEL_READY = True
            print("✓ DistilBART model loaded successfully (fallback)")
            print("="*60 + "\n")
        except Exception as e2:
            print(f"❌ Error loading models: {e2}")
            summarizer = None
            tokenizer = None
            print("="*60 + "\n")


def safe_truncate_text(text, max_tokens=900):
    """Safely truncate text to fit token limit."""
    if not tokenizer:
        # Fallback: estimate 1 token ≈ 4 chars
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            # Take beginning, middle, end
            chunk_size = max_chars // 3
            return text[:chunk_size] + " " + text[len(text)//2:len(text)//2+chunk_size] + " " + text[-chunk_size:]
        return text
    
    try:
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        
        if len(tokens) <= max_tokens:
            return text
        
        print(f"[TRUNCATE] Reducing {len(tokens)} tokens to {max_tokens}")
        
        # Smart chunking
        chunk1_size = int(max_tokens * 0.4)
        chunk2_size = int(max_tokens * 0.3)
        chunk3_size = int(max_tokens * 0.3)
        
        chunk1 = tokens[:chunk1_size]
        middle_start = max(chunk1_size, len(tokens)//2 - chunk2_size//2)
        chunk2 = tokens[middle_start:middle_start+chunk2_size]
        chunk3 = tokens[-chunk3_size:]
        
        # Combine and decode
        combined = chunk1 + chunk2 + chunk3
        result = tokenizer.decode(combined, skip_special_tokens=True)
        
        print(f"[TRUNCATE] Result: ~{len(tokenizer.encode(result))} tokens")
        return result
        
    except Exception as e:
        print(f"[TRUNCATE] Error: {e}, using character-based fallback")
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            chunk_size = max_chars // 3
            return text[:chunk_size] + " " + text[len(text)//2:len(text)//2+chunk_size] + " " + text[-chunk_size:]
        return text


def summarize_text(text, max_length=400, min_length=150):
    """
    Enhanced text summarization with proper token handling.
    """
    if not MODEL_READY:
        load_model()
    
    if not text or len(text.strip()) < 100:
        return "Text too short to summarize."
    
    if not MODEL_READY or summarizer is None:
        return "Summarization model not available."
    
    try:
        print(f"\n[SUMMARIZER] Input: {len(text)} characters")
        
        # Clean text
        cleaned_text = clean_text(text)
        
        # CRITICAL: Truncate to safe token limit (900 tokens for BART's 1024 limit)
        cleaned_text = safe_truncate_text(cleaned_text, max_tokens=900)
        
        if len(cleaned_text.strip()) < 50:
            print("[SUMMARIZER] Content too short after truncation")
            return extractive_summary(text)
        
        # Adaptive summary length
        word_count = len(cleaned_text.split())
        
        if word_count > 600:
            adjusted_max = 450
            adjusted_min = 200
        elif word_count > 400:
            adjusted_max = 350
            adjusted_min = 150
        elif word_count > 200:
            adjusted_max = 250
            adjusted_min = 100
        else:
            adjusted_max = 200
            adjusted_min = 80
        
        adjusted_max = max(adjusted_max, adjusted_min + 50)
        
        print(f"[SUMMARIZER] Generating summary (max={adjusted_max}, min={adjusted_min})")
        
        # Generate summary
        summary = summarizer(
            cleaned_text,
            max_length=adjusted_max,
            min_length=adjusted_min,
            do_sample=False,
            truncation=True,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        result = summary[0]['summary_text']
        print(f"[SUMMARIZER] Generated: {len(result)} chars")
        
        return post_process_summary(result)
    
    except Exception as e:
        print(f"[SUMMARIZER] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return extractive_summary(text)


def summarize_reviews_with_analysis(reviews_data, max_reviews=100):
    """
    Enhanced review analysis with proper token handling.
    """
    if not MODEL_READY:
        load_model()
    
    if not reviews_data:
        return {
            'success': False,
            'brief_summary': "No reviews available to analyze.",
            'detailed_summary': "No reviews available to analyze.",
            'pros': [],
            'cons': [],
            'stats': {}
        }
    
    if not MODEL_READY or summarizer is None:
        return {
            'success': False,
            'brief_summary': "Summarization model not available.",
            'detailed_summary': "Summarization model not available.",
            'pros': [],
            'cons': [],
            'stats': {}
        }
    
    try:
        print(f"\n[REVIEW ANALYZER] Processing {len(reviews_data)} reviews")
        
        reviews_data = reviews_data[:max_reviews]
        
        # Separate reviews
        positive_reviews = []
        negative_reviews = []
        ratings = []
        
        for review in reviews_data:
            rating = review.get('rating')
            body = review.get('body', '')
            title = review.get('title', '')
            full_text = f"{title} {body}".strip()
            
            if not full_text or len(full_text) < 20:
                continue
            
            if rating:
                ratings.append(rating)
                if rating >= 4:
                    positive_reviews.append(full_text)
                elif rating <= 2:
                    negative_reviews.append(full_text)
        
        print(f"[REVIEW ANALYZER] Positive: {len(positive_reviews)}, Negative: {len(negative_reviews)}")
        
        stats = calculate_review_stats(ratings, len(reviews_data))
        
        # Generate summaries
        all_texts = [r.get('body', '') for r in reviews_data if r.get('body')]
        
        brief_summary = generate_brief_summary(all_texts[:30], stats)
        detailed_summary = generate_detailed_summary(all_texts[:50])
        pros = extract_pros(positive_reviews[:40])
        cons = extract_cons(negative_reviews[:40])
        
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


def generate_brief_summary(review_texts, stats):
    """Generate brief 2-3 sentence summary."""
    if not review_texts:
        return "No reviews available."
    
    try:
        combined = " ".join([clean_text(r) for r in review_texts])
        combined = safe_truncate_text(combined, max_tokens=500)
        
        summary = summarizer(
            combined,
            max_length=100,
            min_length=40,
            do_sample=False,
            truncation=True,
            num_beams=4,
            early_stopping=True
        )
        
        brief = summary[0]['summary_text']
        brief = post_process_summary(brief)
        
        if stats.get('average_rating'):
            avg = stats['average_rating']
            sentiment = "highly positive" if avg >= 4.5 else "positive" if avg >= 4.0 else "mixed" if avg >= 3.0 else "negative"
            brief = f"Overall, customers have a {sentiment} experience (avg {avg}/5). " + brief
        
        return brief
    except Exception as e:
        print(f"[BRIEF] Error: {str(e)}")
        return generate_extractive_brief(review_texts, stats)


def generate_extractive_brief(texts, stats):
    """Fallback brief summary."""
    sentences = []
    for t in texts[:10]:
        sentences.extend([s.strip() for s in re.split(r'[.!?]+', t) if 20 < len(s.strip()) < 150])
    
    if sentences:
        brief = ". ".join(sentences[:3]) + "."
        if stats.get('average_rating'):
            avg = stats['average_rating']
            sentiment = "highly positive" if avg >= 4.5 else "positive" if avg >= 4.0 else "mixed"
            brief = f"Overall, customers have a {sentiment} experience (avg {avg}/5). " + brief
        return brief
    return "Reviews available but summary could not be generated."


def generate_detailed_summary(review_texts):
    """Generate detailed summary."""
    if not review_texts:
        return "No reviews available."
    
    try:
        combined = " ".join([clean_text(r) for r in review_texts])
        combined = safe_truncate_text(combined, max_tokens=800)
        
        summary = summarizer(
            combined,
            max_length=350,
            min_length=150,
            do_sample=False,
            truncation=True,
            num_beams=4,
            early_stopping=True
        )
        
        return post_process_summary(summary[0]['summary_text'])
    except Exception as e:
        print(f"[DETAILED] Error: {str(e)}")
        return extractive_summary(review_texts, 8)


def extract_pros(positive_reviews):
    """Extract pros."""
    if not positive_reviews or len(positive_reviews) < 3:
        return []
    
    try:
        combined = " ".join([clean_text(r) for r in positive_reviews])
        combined = safe_truncate_text(combined, max_tokens=600)
        
        summary = summarizer(
            combined,
            max_length=180,
            min_length=60,
            do_sample=False,
            truncation=True,
            num_beams=4
        )
        
        text = summary[0]['summary_text']
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 15]
        return sentences[:5]
    except:
        return extract_keyword_pros(positive_reviews)


def extract_cons(negative_reviews):
    """Extract cons."""
    if not negative_reviews or len(negative_reviews) < 3:
        return []
    
    try:
        combined = " ".join([clean_text(r) for r in negative_reviews])
        combined = safe_truncate_text(combined, max_tokens=600)
        
        summary = summarizer(
            combined,
            max_length=180,
            min_length=60,
            do_sample=False,
            truncation=True,
            num_beams=4
        )
        
        text = summary[0]['summary_text']
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 15]
        return sentences[:5]
    except:
        return extract_keyword_cons(negative_reviews)


def extract_keyword_pros(reviews):
    """Fallback pros."""
    combined = " ".join(reviews[:20]).lower()
    pros = []
    
    patterns = [
        (r'excellent\s+(\w+)', 'Excellent {}'),
        (r'great\s+(\w+)', 'Great {}'),
        (r'love\s+the\s+(\w+)', 'Users love the {}'),
    ]
    
    for pattern, template in patterns:
        matches = re.findall(pattern, combined)
        if matches and len(pros) < 5:
            pros.append(template.format(matches[0]))
    
    return pros[:5] if pros else ["Customers generally satisfied"]


def extract_keyword_cons(reviews):
    """Fallback cons."""
    combined = " ".join(reviews[:20]).lower()
    cons = []
    
    patterns = [
        (r'poor\s+(\w+)', 'Poor {}'),
        (r'bad\s+(\w+)', 'Bad {}'),
        (r'(\w+)\s+issue', '{} issues'),
    ]
    
    for pattern, template in patterns:
        matches = re.findall(pattern, combined)
        if matches and len(cons) < 5:
            cons.append(template.format(matches[0]))
    
    return cons[:5] if cons else ["Some customers reported issues"]


def calculate_review_stats(ratings, total):
    """Calculate stats."""
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
        if r >= 4.5: stats['rating_distribution']['5_star'] += 1
        elif r >= 3.5: stats['rating_distribution']['4_star'] += 1
        elif r >= 2.5: stats['rating_distribution']['3_star'] += 1
        elif r >= 1.5: stats['rating_distribution']['2_star'] += 1
        else: stats['rating_distribution']['1_star'] += 1
    
    pos = stats['rating_distribution']['5_star'] + stats['rating_distribution']['4_star']
    neg = stats['rating_distribution']['1_star'] + stats['rating_distribution']['2_star']
    
    stats['positive_percentage'] = round((pos / len(ratings)) * 100, 1)
    stats['negative_percentage'] = round((neg / len(ratings)) * 100, 1)
    
    return stats


def clean_text(text):
    """Clean text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:()\-\'\""]', '', text)
    return text.strip()


def post_process_summary(summary):
    """Post-process."""
    if not summary.endswith(('.', '!', '?')):
        summary += '.'
    summary = re.sub(r'\s+', ' ', summary).strip()
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    return summary


def extractive_summary(text, num_sentences=8):
    """Fallback extractive."""
    if isinstance(text, list):
        sentences = []
        for t in text[:15]:
            sentences.extend([s.strip() for s in re.split(r'[.!?]+', t) if len(s.strip()) > 20])
    else:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
    
    if sentences:
        n = min(num_sentences, len(sentences))
        return ". ".join(sentences[:n]) + "."
    return "Content available but summary could not be generated."