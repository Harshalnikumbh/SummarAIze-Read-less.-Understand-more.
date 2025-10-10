import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import pipeline
import torch
import re
from collections import Counter

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Model state
MODEL_READY = False
summarizer = None

def load_model():
    """Load the summarization model (call this from Flask app)"""
    global MODEL_READY, summarizer
    
    if MODEL_READY:
        return
    
    print("\n" + "="*60)
    print("📦 Loading summarization model...")
    print("="*60)
    
    try:
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
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
            MODEL_READY = True
            print("✓ DistilBART model loaded successfully (fallback)")
            print("="*60 + "\n")
        except Exception as e2:
            print(f"❌ Error loading models: {e2}")
            summarizer = None
            print("="*60 + "\n")


def summarize_text(text, max_length=350, min_length=150):
    """
    Summarize any text content using BART with enhanced detail.
    
    Args:
        text: Text content to summarize
        max_length: Maximum summary length
        min_length: Minimum summary length
        
    Returns:
        Summary string
    """
    if not MODEL_READY:
        load_model()
    
    if not text or len(text.strip()) < 100:
        return "Text too short to summarize."
    
    if not MODEL_READY or summarizer is None:
        return "Summarization model not available."
    
    try:
        print(f"\n[SUMMARIZER] Input text length: {len(text)} characters")
        
        cleaned_text = clean_text(text)
        print(f"[SUMMARIZER] Cleaned text length: {len(cleaned_text)} characters")
        
        max_input_words = 650
        words = cleaned_text.split()
        print(f"[SUMMARIZER] Word count: {len(words)}")
        
        if len(words) > max_input_words:
            print(f"[SUMMARIZER] Truncating from {len(words)} to {max_input_words} words")
            
            first_size = int(max_input_words * 0.6)
            last_size = int(max_input_words * 0.4)
            
            first_size = min(first_size, len(words))
            last_size = min(last_size, len(words))
            
            first_part = words[:first_size]
            last_part = words[-last_size:] if last_size > 0 else []
            
            words = first_part + last_part
            cleaned_text = " ".join(words)
            print(f"[SUMMARIZER] After truncation: {len(words)} words")
        
        if len(cleaned_text.split()) > max_input_words:
            cleaned_text = " ".join(cleaned_text.split()[:max_input_words])
            print(f"[SUMMARIZER] Safety truncation applied")
        
        if len(cleaned_text.split()) < 50:
            print("[SUMMARIZER] Content too short after cleaning")
            return extractive_summary(text)
        
        word_count = len(words)
        
        if word_count > 600:
            adjusted_max_length = 500
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
        
        adjusted_max_length = max(adjusted_max_length, adjusted_min_length + 50)
        
        print(f"[SUMMARIZER] Generating summary (max={adjusted_max_length}, min={adjusted_min_length})")
        
        summary = summarizer(
            cleaned_text,
            max_length=adjusted_max_length,
            min_length=adjusted_min_length,
            do_sample=False,
            truncation=True,
            max_new_tokens=adjusted_max_length,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        result = summary[0]['summary_text']
        print(f"[SUMMARIZER] Summary generated: {len(result)} characters, {len(result.split())} words")
        
        result = post_process_summary(result)
        
        return result
    
    except Exception as e:
        print(f"[SUMMARIZER] Error during summarization: {str(e)}")
        import traceback
        traceback.print_exc()
        return extractive_summary(text)


def summarize_reviews_with_analysis(reviews_data, max_reviews=50):
    """
    Analyze product reviews and provide summary with pros, cons, and overall sentiment.
    
    Args:
        reviews_data: List of review dictionaries with keys: rating, title, body, author, etc.
        max_reviews: Maximum number of reviews to process
        
    Returns:
        Dictionary containing brief_summary, detailed_summary, pros, cons, and statistics
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
        
        # Limit reviews
        reviews_data = reviews_data[:max_reviews]
        
        # Separate positive and negative reviews based on rating
        positive_reviews = []
        negative_reviews = []
        neutral_reviews = []
        
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
                else:
                    neutral_reviews.append(full_text)
            else:
                # If no rating, use sentiment analysis fallback
                neutral_reviews.append(full_text)
        
        print(f"[REVIEW ANALYZER] Positive: {len(positive_reviews)}, Negative: {len(negative_reviews)}, Neutral: {len(neutral_reviews)}")
        
        # Calculate statistics
        stats = calculate_review_stats(ratings, len(reviews_data))
        
        # Generate overall summaries
        all_review_texts = [r.get('body', '') for r in reviews_data if r.get('body')]
        
        # Generate brief summary (2-3 lines)
        brief_summary = generate_brief_review_summary(all_review_texts, stats)
        
        # Generate detailed summary (original longer version)
        detailed_summary = generate_review_summary(all_review_texts)
        
        # Extract pros from positive reviews
        pros = extract_pros(positive_reviews)
        
        # Extract cons from negative reviews
        cons = extract_cons(negative_reviews)
        
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


def generate_brief_review_summary(review_texts, stats):
    """
    Generate a brief 2-3 line summary of reviews with key insights.
    """
    if not review_texts:
        return "No reviews available."
    
    try:
        # Combine reviews
        combined_text = " ".join([clean_text(r) for r in review_texts[:20]])  # Use fewer reviews for brief summary
        
        # Limit to smaller input for concise output
        words = combined_text.split()
        if len(words) > 400:
            words = words[:400]
            combined_text = " ".join(words)
        
        # Generate very concise summary
        summary = summarizer(
            combined_text,
            max_length=100,  # Much shorter - about 2-3 sentences
            min_length=40,   # Minimum 40 tokens
            do_sample=False,
            truncation=True,
            max_new_tokens=100,
            num_beams=4,
            length_penalty=0.8,  # Slightly favor shorter summaries
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        brief_text = summary[0]['summary_text']
        brief_text = post_process_summary(brief_text)
        
        # Add rating context if available
        if stats.get('average_rating'):
            avg_rating = stats['average_rating']
            sentiment = "highly positive" if avg_rating >= 4.5 else "positive" if avg_rating >= 4.0 else "mixed" if avg_rating >= 3.0 else "negative"
            
            # Prepend rating context
            rating_intro = f"Overall, customers have a {sentiment} experience (avg {avg_rating}/5). "
            brief_text = rating_intro + brief_text
        
        print(f"[BRIEF SUMMARY] Generated: {len(brief_text)} characters")
        return brief_text
    
    except Exception as e:
        print(f"[BRIEF SUMMARY] Error: {str(e)}")
        # Fallback to extractive brief summary
        return generate_extractive_brief_summary(review_texts, stats)


def generate_extractive_brief_summary(review_texts, stats):
    """Fallback extractive brief summary (2-3 sentences)."""
    if not review_texts:
        return "No reviews available."
    
    # Get most representative sentences
    all_sentences = []
    for review in review_texts[:10]:
        sentences = re.split(r'[.!?]+', review)
        all_sentences.extend([s.strip() for s in sentences if 20 < len(s.strip()) < 150])
    
    if all_sentences:
        # Take first 2-3 sentences
        summary_sentences = all_sentences[:3]
        brief = ". ".join(summary_sentences) + "."
        
        # Add rating context
        if stats.get('average_rating'):
            avg_rating = stats['average_rating']
            sentiment = "highly positive" if avg_rating >= 4.5 else "positive" if avg_rating >= 4.0 else "mixed" if avg_rating >= 3.0 else "negative"
            rating_intro = f"Overall, customers have a {sentiment} experience (avg {avg_rating}/5). "
            brief = rating_intro + brief
        
        return brief
    
    return "Reviews available but brief summary could not be generated."


def generate_brief_review_summary(review_texts, stats):
    """
    Generate a brief 2-3 line summary of reviews with key insights.
    """
    if not review_texts:
        return "No reviews available."
    
    try:
        # Combine reviews
        combined_text = " ".join([clean_text(r) for r in review_texts[:20]])  # Use fewer reviews for brief summary
        
        # Limit to smaller input for concise output
        words = combined_text.split()
        if len(words) > 400:
            words = words[:400]
            combined_text = " ".join(words)
        
        # Generate very concise summary
        summary = summarizer(
            combined_text,
            max_length=100,  # Much shorter - about 2-3 sentences
            min_length=40,   # Minimum 40 tokens
            do_sample=False,
            truncation=True,
            max_new_tokens=100,
            num_beams=4,
            length_penalty=0.8,  # Slightly favor shorter summaries
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        brief_text = summary[0]['summary_text']
        brief_text = post_process_summary(brief_text)
        
        # Add rating context if available
        if stats.get('average_rating'):
            avg_rating = stats['average_rating']
            sentiment = "highly positive" if avg_rating >= 4.5 else "positive" if avg_rating >= 4.0 else "mixed" if avg_rating >= 3.0 else "negative"
            
            # Prepend rating context
            rating_intro = f"Overall, customers have a {sentiment} experience (avg {avg_rating}/5). "
            brief_text = rating_intro + brief_text
        
        print(f"[BRIEF SUMMARY] Generated: {len(brief_text)} characters")
        return brief_text
    
    except Exception as e:
        print(f"[BRIEF SUMMARY] Error: {str(e)}")
        # Fallback to extractive brief summary
        return generate_extractive_brief_summary(review_texts, stats)


def generate_extractive_brief_summary(review_texts, stats):
    """Fallback extractive brief summary (2-3 sentences)."""
    if not review_texts:
        return "No reviews available."
    
    # Get most representative sentences
    all_sentences = []
    for review in review_texts[:10]:
        sentences = re.split(r'[.!?]+', review)
        all_sentences.extend([s.strip() for s in sentences if 20 < len(s.strip()) < 150])
    
    if all_sentences:
        # Take first 2-3 sentences
        summary_sentences = all_sentences[:3]
        brief = ". ".join(summary_sentences) + "."
        
        # Add rating context
        if stats.get('average_rating'):
            avg_rating = stats['average_rating']
            sentiment = "highly positive" if avg_rating >= 4.5 else "positive" if avg_rating >= 4.0 else "mixed" if avg_rating >= 3.0 else "negative"
            rating_intro = f"Overall, customers have a {sentiment} experience (avg {avg_rating}/5). "
            brief = rating_intro + brief
        
        return brief
    
    return "Reviews available but brief summary could not be generated."


def generate_review_summary(review_texts):
    """Generate an overall summary of all reviews."""
    if not review_texts:
        return "No reviews available."
    
    try:
        combined_text = " ".join([clean_text(r) for r in review_texts])
        
        max_input_words = 600
        words = combined_text.split()
        
        if len(words) > max_input_words:
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
        
        summary = summarizer(
            combined_text,
            max_length=250,
            min_length=100,
            do_sample=False,
            truncation=True,
            max_new_tokens=250,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        result = summary[0]['summary_text']
        return post_process_summary(result)
    
    except Exception as e:
        print(f"[REVIEW SUMMARY] Error: {str(e)}")
        return extractive_summary(review_texts, num_sentences=6)


def extract_pros(positive_reviews):
    """
    Extract pros/positive points from positive reviews.
    """
    if not positive_reviews:
        return []
    
    print(f"[PROS EXTRACTION] Analyzing {len(positive_reviews)} positive reviews")
    
    try:
        # Combine positive reviews
        combined_positive = " ".join([clean_text(r) for r in positive_reviews[:30]])
        
        # Limit input size
        words = combined_positive.split()
        if len(words) > 500:
            words = words[:500]
            combined_positive = " ".join(words)
        
        # Generate summary focused on positive aspects
        summary = summarizer(
            combined_positive,
            max_length=200,
            min_length=80,
            do_sample=False,
            truncation=True,
            max_new_tokens=200,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        pros_text = summary[0]['summary_text']
        
        # Split into bullet points
        pros_sentences = re.split(r'[.!?]+', pros_text)
        pros_list = [s.strip() for s in pros_sentences if len(s.strip()) > 15]
        
        # Also extract common positive keywords
        positive_keywords = extract_common_themes(positive_reviews, positive=True)
        
        # Combine and deduplicate
        pros_final = []
        for pro in pros_list[:5]:  # Max 5 pros
            if pro and pro not in pros_final:
                pros_final.append(pro)
        
        # Add keyword-based pros if needed
        for keyword in positive_keywords[:3]:
            keyword_pro = f"Users appreciate the {keyword}"
            if keyword_pro not in ' '.join(pros_final).lower() and len(pros_final) < 5:
                pros_final.append(keyword_pro)
        
        return pros_final[:5]  # Return max 5 pros
    
    except Exception as e:
        print(f"[PROS EXTRACTION] Error: {str(e)}")
        return extract_keyword_based_pros(positive_reviews)


def extract_cons(negative_reviews):
    """
    Extract cons/negative points from negative reviews.
    """
    if not negative_reviews:
        return []
    
    print(f"[CONS EXTRACTION] Analyzing {len(negative_reviews)} negative reviews")
    
    try:
        # Combine negative reviews
        combined_negative = " ".join([clean_text(r) for r in negative_reviews[:30]])
        
        # Limit input size
        words = combined_negative.split()
        if len(words) > 500:
            words = words[:500]
            combined_negative = " ".join(words)
        
        # Generate summary focused on negative aspects
        summary = summarizer(
            combined_negative,
            max_length=200,
            min_length=80,
            do_sample=False,
            truncation=True,
            max_new_tokens=200,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        cons_text = summary[0]['summary_text']
        
        # Split into bullet points
        cons_sentences = re.split(r'[.!?]+', cons_text)
        cons_list = [s.strip() for s in cons_sentences if len(s.strip()) > 15]
        
        # Extract common negative keywords
        negative_keywords = extract_common_themes(negative_reviews, positive=False)
        
        # Combine and deduplicate
        cons_final = []
        for con in cons_list[:5]:  # Max 5 cons
            if con and con not in cons_final:
                cons_final.append(con)
        
        # Add keyword-based cons if needed
        for keyword in negative_keywords[:3]:
            keyword_con = f"Users complain about {keyword}"
            if keyword_con not in ' '.join(cons_final).lower() and len(cons_final) < 5:
                cons_final.append(keyword_con)
        
        return cons_final[:5]  # Return max 5 cons
    
    except Exception as e:
        print(f"[CONS EXTRACTION] Error: {str(e)}")
        return extract_keyword_based_cons(negative_reviews)


def extract_common_themes(reviews, positive=True):
    """
    Extract common themes/keywords from reviews.
    """
    # Positive and negative indicator words
    positive_words = [
        'great', 'excellent', 'good', 'amazing', 'love', 'perfect', 'best',
        'quality', 'fast', 'easy', 'comfortable', 'durable', 'recommended',
        'worth', 'happy', 'satisfied', 'awesome', 'fantastic', 'wonderful'
    ]
    
    negative_words = [
        'bad', 'poor', 'worst', 'terrible', 'awful', 'hate', 'disappointing',
        'defective', 'broken', 'issue', 'problem', 'slow', 'difficult',
        'uncomfortable', 'cheap', 'waste', 'unhappy', 'disappointed', 'fake'
    ]
    
    # Product feature keywords
    feature_words = [
        'quality', 'price', 'value', 'design', 'build', 'performance',
        'battery', 'camera', 'screen', 'sound', 'display', 'delivery',
        'packaging', 'size', 'fit', 'color', 'material', 'durability',
        'speed', 'customer service', 'warranty', 'features'
    ]
    
    combined_text = " ".join(reviews).lower()
    
    # Count feature mentions with sentiment words nearby
    theme_counts = Counter()
    
    for feature in feature_words:
        # Check if feature appears with sentiment words
        pattern = r'\b\w+\s+' + re.escape(feature) + r'|\b' + re.escape(feature) + r'\s+\w+\b'
        matches = re.findall(pattern, combined_text)
        
        for match in matches:
            sentiment_words = positive_words if positive else negative_words
            if any(word in match for word in sentiment_words):
                theme_counts[feature] += 1
    
    # Return top themes
    return [theme for theme, count in theme_counts.most_common(5)]


def extract_keyword_based_pros(positive_reviews):
    """Fallback method for extracting pros using keyword analysis."""
    pros = []
    
    positive_patterns = [
        (r'(great|excellent|good|amazing)\s+(\w+)', 'positive attribute'),
        (r'love\s+the\s+(\w+)', 'loved feature'),
        (r'(\w+)\s+is\s+(perfect|excellent|great)', 'quality aspect'),
        (r'highly\s+recommend', 'Highly recommended by users'),
        (r'worth\s+the\s+(price|money)', 'Good value for money')
    ]
    
    combined = " ".join(positive_reviews).lower()
    
    for pattern, template in positive_patterns:
        matches = re.findall(pattern, combined)
        if matches and len(pros) < 5:
            if isinstance(matches[0], tuple):
                feature = matches[0][0] if matches[0][0] else matches[0][1]
            else:
                feature = matches[0]
            
            if template == 'positive attribute':
                pros.append(f"Excellent {feature}")
            elif template == 'loved feature':
                pros.append(f"Users love the {feature}")
            elif template == 'quality aspect':
                pros.append(f"High quality {feature}")
            else:
                pros.append(template)
    
    return pros[:5]


def extract_keyword_based_cons(negative_reviews):
    """Fallback method for extracting cons using keyword analysis."""
    cons = []
    
    negative_patterns = [
        (r'(poor|bad|terrible)\s+(\w+)', 'negative attribute'),
        (r'(\w+)\s+(issue|problem|defect)', 'problematic feature'),
        (r'disappointed\s+with\s+(\w+)', 'disappointing aspect'),
        (r'waste\s+of\s+money', 'Not worth the price'),
        (r'do\s+not\s+buy', 'Not recommended by users')
    ]
    
    combined = " ".join(negative_reviews).lower()
    
    for pattern, template in negative_patterns:
        matches = re.findall(pattern, combined)
        if matches and len(cons) < 5:
            if isinstance(matches[0], tuple):
                feature = matches[0][0] if matches[0][0] else matches[0][1]
            else:
                feature = matches[0]
            
            if template == 'negative attribute':
                cons.append(f"Poor {feature}")
            elif template == 'problematic feature':
                cons.append(f"Issues with {feature}")
            elif template == 'disappointing aspect':
                cons.append(f"Disappointing {feature}")
            else:
                cons.append(template)
    
    return cons[:5]


def calculate_review_stats(ratings, total_reviews):
    """Calculate statistics from review ratings."""
    stats = {
        'total_reviews': total_reviews,
        'average_rating': 0.0,
        'rating_distribution': {
            '5_star': 0,
            '4_star': 0,
            '3_star': 0,
            '2_star': 0,
            '1_star': 0
        },
        'positive_percentage': 0.0,
        'negative_percentage': 0.0
    }
    
    if not ratings:
        return stats
    
    # Calculate average
    stats['average_rating'] = round(sum(ratings) / len(ratings), 2)
    
    # Calculate distribution
    for rating in ratings:
        if rating >= 4.5:
            stats['rating_distribution']['5_star'] += 1
        elif rating >= 3.5:
            stats['rating_distribution']['4_star'] += 1
        elif rating >= 2.5:
            stats['rating_distribution']['3_star'] += 1
        elif rating >= 1.5:
            stats['rating_distribution']['2_star'] += 1
        else:
            stats['rating_distribution']['1_star'] += 1
    
    # Calculate percentages
    positive_count = stats['rating_distribution']['5_star'] + stats['rating_distribution']['4_star']
    negative_count = stats['rating_distribution']['1_star'] + stats['rating_distribution']['2_star']
    
    stats['positive_percentage'] = round((positive_count / len(ratings)) * 100, 1) if ratings else 0
    stats['negative_percentage'] = round((negative_count / len(ratings)) * 100, 1) if ratings else 0
    
    return stats


def clean_text(text):
    """Clean text for summarization while preserving important content."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:()\-\'\""]', '', text)
    text = ' '.join([w for w in text.split() if len(w) > 1 or w in '.,!?'])
    return text.strip()


def post_process_summary(summary):
    """Post-process the summary to ensure proper formatting and readability."""
    if not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    summary = re.sub(r'([.!?])([A-Z])', r'\1 \2', summary)
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    return summary


def extractive_summary(text, num_sentences=8):
    """Enhanced fallback extractive summary."""
    if isinstance(text, list):
        all_sentences = []
        for review in text[:15]:
            sentences = re.split(r'[.!?]+', review)
            all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 20])
    else:
        sentences = re.split(r'[.!?]+', text)
        all_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if all_sentences:
        num_sentences = min(num_sentences, len(all_sentences))
        
        if len(all_sentences) <= num_sentences:
            summary_sentences = all_sentences
        else:
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


# Maintain backward compatibility
def summarize_reviews(review_texts, max_reviews=50):
    """
    Legacy function for basic review summarization (without pros/cons).
    For new code, use summarize_reviews_with_analysis instead.
    """
    if not MODEL_READY:
        load_model()
    
    if not review_texts:
        return "No reviews available to summarize."
    
    if not MODEL_READY or summarizer is None:
        return "Summarization model not available."
    
    return generate_review_summary(review_texts)