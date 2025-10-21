import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re
import time
from urllib.parse import urlparse, parse_qs


class WebpageScraper:
    """
    Enhanced webpage scraper with improved review quality and noise filtering.
    Now includes better content type detection and metadata extraction.
    """

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()

    def scrape(self, url: str, max_pages: int = 8) -> Dict:
        """
        Intelligently scrape webpage - detects if it's a review page or regular webpage.
        Returns content_type for better summarization.
        """
        try:
            print(f"\n[SCRAPER] Fetching URL: {url}")
            response = self.session.get(url, headers=self.headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            print(f"[SCRAPER] Response status: {response.status_code}")

            soup = BeautifulSoup(response.content, 'html.parser')

            url_type = self._detect_url_type(url, soup)
            print(f"[SCRAPER] URL Type: {url_type}")

            self._remove_noise(soup)

            title = self._get_title(soup)
            content = self._get_content(soup)
            metadata = self._get_metadata(soup, url)

            # NEW: Map URL type to content type for summarizer
            content_type = self._map_url_type_to_content_type(url_type, content, url)

            result = {
                'success': True,
                'title': title,
                'content': content,
                'metadata': metadata,
                'url': url,
                'url_type': url_type,
                'content_type': content_type,  # NEW: For summarizer
                'reviews': []
            }

            if url_type in ['review_page', 'product_page']:
                reviews = self._scrape_reviews_with_pagination(soup, url, max_pages)
                reviews = self._clean_and_deduplicate_reviews(reviews)
                result['reviews'] = reviews
                print(f"[SCRAPER] Extracted {len(reviews)} clean, unique reviews")

            print(f"[SCRAPER] Title: {title}")
            print(f"[SCRAPER] Content length: {len(content)} characters")
            print(f"[SCRAPER] Content Type: {content_type}")

            return result

        except requests.exceptions.HTTPError as e:
            return {
                'success': False,
                'error': f'HTTP Error {e.response.status_code}',
                'status_code': e.response.status_code,
                'url': url
            }
        except Exception as e:
            print(f"[SCRAPER] Error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    def _map_url_type_to_content_type(self, url_type: str, content: str, url: str) -> str:
        """
        NEW: Map URL type to content type for better summarization.
        Uses URL patterns and content analysis.
        """
        if url_type in ['review_page', 'product_page']:
            return 'product_review'
        
        # For regular webpages, analyze URL and content
        url_lower = url.lower()
        
        # Check for news domains and patterns
        news_indicators = [
            'news', 'article', 'story', 'breaking', 'report',
            'bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com',
            'theguardian.com', 'timesofindia.com', 'hindustantimes.com',
            'indianexpress.com', 'ndtv.com', 'thehindu.com',
            '/article/', '/news/', '/story/', '/breaking-'
        ]
        
        if any(indicator in url_lower for indicator in news_indicators):
            print(f"[SCRAPER] News article detected from URL")
            return 'news_article'
        
        # Check for technical documentation patterns
        tech_indicators = [
            'docs', 'documentation', 'api', 'guide', 'tutorial',
            'github.com', 'readthedocs', 'developer',
            '/docs/', '/api/', '/reference/', '/guide/'
        ]
        
        if any(indicator in url_lower for indicator in tech_indicators):
            print(f"[SCRAPER] Technical documentation detected from URL")
            return 'technical_doc'
        
        # Analyze content if URL doesn't give clear signal
        return self._detect_content_type_from_text(content)

    def _detect_content_type_from_text(self, text: str) -> str:
        """
        NEW: Quick content type detection from text (lightweight version).
        """
        if not text or len(text) < 100:
            return 'general_webpage'
        
        text_lower = text.lower()
        
        # News keywords
        news_score = sum(1 for kw in ['reported', 'according to', 'officials', 
                                       'statement', 'sources', 'announced', 'confirmed']
                        if kw in text_lower)
        
        # Technical keywords
        tech_score = sum(1 for kw in ['function', 'api', 'parameter', 'method',
                                       'installation', 'configuration', 'syntax']
                        if kw in text_lower)
        
        if news_score >= 3:
            return 'news_article'
        elif tech_score >= 3:
            return 'technical_doc'
        
        return 'general_webpage'

    def _clean_and_deduplicate_reviews(self, reviews: List[Dict]) -> List[Dict]:
        """Remove duplicates and clean review data."""
        seen_bodies = set()
        cleaned_reviews = []
        
        for review in reviews:
            body = review.get('body', '').strip()
            
            if not body or len(body) < 30:
                continue
            
            body = self._clean_review_text(body)
            
            if body in seen_bodies:
                continue
            
            if self._is_near_duplicate(body, seen_bodies):
                continue
            
            seen_bodies.add(body)
            review['body'] = body
            
            review['title'] = review.get('title', '').strip()
            review['author'] = review.get('author', 'Anonymous').strip()
            review['date'] = review.get('date', '').strip()
            
            rating = review.get('rating', 3.0)
            if not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
                rating = 3.0
            review['rating'] = float(rating)
            
            cleaned_reviews.append(review)
        
        return cleaned_reviews

    def _clean_review_text(self, text: str) -> str:
        """Clean review text by removing noise patterns."""
        noise_patterns = [
            r'READ MORE',
            r'Read More',
            r'Show More',
            r'Certified Buyer',
            r'\d+\s+people found this helpful',
            r'Was this review helpful\?',
            r'Report Abuse',
            r'👍\s*\d*',
            r'👎\s*\d*',
            r'Most Helpful',
            r'Most Recent',
            r'Positive First',
            r'Negative First',
            r'Overall\s+Picture\s+Sound.*?App Support',
            r'flipkart\.com',
            r'amazon\.com',
            r'myntra\.com',
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()

    def _is_near_duplicate(self, text: str, seen_texts: set, threshold: float = 0.8) -> bool:
        """Check if text is near-duplicate of any seen text."""
        text_words = set(text.lower().split())
        
        for seen in seen_texts:
            seen_words = set(seen.lower().split())
            
            if len(text_words) == 0 or len(seen_words) == 0:
                continue
            
            intersection = len(text_words & seen_words)
            union = len(text_words | seen_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= threshold:
                return True
        
        return False

    def _detect_url_type(self, url: str, soup: BeautifulSoup) -> str:
        """Detect if URL is a review page, product page, or regular webpage."""
        url_lower = url.lower()

        review_page_patterns = [
            r'/product-reviews/',
            r'/customer-reviews/',
            r'/reviews/',
        ]

        for pattern in review_page_patterns:
            if re.search(pattern, url_lower):
                print(f"[DETECTION] Review page detected by URL pattern: {pattern}")
                return 'review_page'

        product_patterns = [
            r'amazon\.(com|in|co\.uk)/.*/dp/',
            r'flipkart\.com/.*/p/',
            r'myntra\.com/.*-\d+/buy',
            r'/product/',
            r'/item/',
        ]

        for pattern in product_patterns:
            if re.search(pattern, url_lower):
                print(f"[DETECTION] Product page detected by URL pattern: {pattern}")
                return 'product_page'

        product_schema = soup.find(attrs={'itemtype': re.compile(r'Product', re.IGNORECASE)})
        if product_schema:
            print(f"[DETECTION] Product page detected by schema")
            return 'product_page'

        print(f"[DETECTION] Regular webpage detected")
        return 'webpage'

    def _scrape_reviews_with_pagination(self, soup: BeautifulSoup, url: str, max_pages: int = 8) -> List[Dict]:
        """Scrape product reviews with enhanced pagination support."""
        all_reviews = []

        if 'flipkart' in url.lower():
            all_reviews = self._scrape_flipkart_with_pagination(url, max_pages)
        elif 'amazon' in url.lower():
            all_reviews = self._scrape_amazon_with_pagination(url, max_pages)
        elif 'myntra' in url.lower():
            all_reviews = self._scrape_myntra_reviews(soup)
        else:
            all_reviews = self._scrape_generic_reviews(soup)

        return all_reviews

    def _scrape_flipkart_with_pagination(self, url: str, max_pages: int = 8) -> List[Dict]:
        """Flipkart pagination with HTML mode."""
        print("[FLIPKART] Using HTML mode")
        all_reviews = []
        
        for page in range(1, max_pages + 1):
            page_url = f"{url}&page={page}" if "?" in url else f"{url}?page={page}"
            
            try:
                print(f"[FLIPKART] Fetching page {page}: {page_url}")
                response = self.session.get(page_url, headers=self.headers, timeout=15)
                
                if response.status_code != 200:
                    print(f"[FLIPKART] Page {page} returned {response.status_code}, stopping.")
                    break
                
                soup = BeautifulSoup(response.content, "html.parser")
                page_reviews = self._parse_flipkart_reviews(soup)
                
                if not page_reviews:
                    print(f"[FLIPKART] No reviews found on page {page}, stopping.")
                    break
                
                all_reviews.extend(page_reviews)
                print(f"[FLIPKART] Page {page}: Found {len(page_reviews)} reviews")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"[FLIPKART] Error on page {page}: {e}")
                break
        
        print(f"[FLIPKART] ✅ Total extracted: {len(all_reviews)} reviews")
        return all_reviews

    def _parse_flipkart_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Enhanced Flipkart review parser with better filtering."""
        reviews = []
        print(f"[FLIPKART] Parsing reviews...")

        strategies = [
            lambda: soup.find_all('div', class_=lambda x: x and ('col' in str(x) or 'review' in str(x).lower())),
            lambda: soup.find_all('div', text=re.compile(r'.{50,}', re.DOTALL)),
            lambda: soup.find_all('div', recursive=True),
        ]

        all_containers = []
        for strategy_func in strategies:
            try:
                containers = strategy_func()
                if containers:
                    all_containers.extend(containers[:200])
                    if len(all_containers) > 50:
                        break
            except Exception:
                continue

        print(f"[FLIPKART] Checking {len(all_containers)} containers")

        seen_texts = set()
        for container in all_containers:
            try:
                full_text = container.get_text(separator=' ', strip=True)
                
                if len(full_text) < 50 or len(full_text) > 3000:
                    continue
                
                if full_text in seen_texts:
                    continue
                
                if not self._is_review_text(full_text):
                    continue
                
                seen_texts.add(full_text)
                
                review_data = self._extract_flipkart_review_smart(container)
                
                if review_data and review_data.get('body') and len(review_data['body']) > 30:
                    reviews.append(review_data)
                
                if len(reviews) >= 50:
                    break
                    
            except Exception:
                continue

        print(f"[FLIPKART] Extracted {len(reviews)} valid reviews from this page")
        return reviews

    def _extract_flipkart_review_smart(self, container) -> Optional[Dict]:
        """Smart extraction for Flipkart reviews."""
        full_text = container.get_text(separator=' ', strip=True)
        
        if len(full_text) < 50:
            return None

        rating = None
        
        rating_elements = container.find_all(['div', 'span'], class_=re.compile(r'star|rating|_3LWZlK', re.I))
        for elem in rating_elements:
            text = elem.get_text(strip=True)
            match = re.search(r'(\d+(?:\.\d+)?)', text)
            if match:
                rating = float(match.group(1))
                if 1 <= rating <= 5:
                    break

        if not rating:
            star_count = full_text.count('★') + full_text.count('⭐')
            if 1 <= star_count <= 5:
                rating = float(star_count)

        if not rating:
            rating = self._infer_rating_from_text(full_text)

        body = self._extract_review_body(full_text)
        
        if not body or len(body) < 30:
            return None

        author = "Anonymous"
        author_patterns = [
            r'(?:by|By|BY)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'Certified Buyer.*?([A-Z][a-z]+)',
        ]
        for pattern in author_patterns:
            match = re.search(pattern, full_text)
            if match:
                potential_author = match.group(1).strip()
                if potential_author not in ['Certified', 'Buyer', 'Review', 'Rating']:
                    author = potential_author
                    break

        date = ""
        date_match = re.search(
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
            full_text,
            re.I
        )
        if date_match:
            date = date_match.group(1)

        return {
            'rating': rating,
            'title': "",
            'body': body,
            'author': author,
            'date': date,
            'helpful': ""
        }

    def _extract_review_body(self, full_text: str) -> str:
        """Extract clean review body from full text."""
        noise_patterns = [
            r'READ MORE',
            r'Certified Buyer',
            r'\d+\s+people found this helpful',
            r'Was this review helpful\?',
            r'👍\s*\d+',
            r'👎\s*\d+',
            r'Report Abuse',
            r'Most Helpful',
            r'Most Recent',
            r'Positive First',
            r'Negative First',
        ]
        
        cleaned = full_text
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        sentences = re.split(r'[.!?]+', cleaned)
        
        valid_sentences = []
        for s in sentences:
            s = s.strip()
            if 30 < len(s) < 500 and self._is_review_sentence(s):
                valid_sentences.append(s)
        
        if valid_sentences:
            return ". ".join(valid_sentences[:5]) + "."
        
        if len(cleaned) > 30:
            return cleaned[:500].strip()
        
        return ""

    def _is_review_sentence(self, text: str) -> bool:
        """Check if a sentence looks like part of a review."""
        text_lower = text.lower()
        
        review_words = [
            'product', 'quality', 'good', 'bad', 'excellent', 'poor',
            'buy', 'bought', 'purchase', 'order', 'delivery', 'price',
            'use', 'using', 'used', 'work', 'works', 'phone', 'camera',
            'battery', 'screen', 'display', 'performance', 'recommend',
            'satisfied', 'disappointed', 'love', 'like', 'hate', 'worth',
            'value', 'money', 'fast', 'slow', 'great', 'nice', 'awesome'
        ]
        
        return any(word in text_lower for word in review_words)

    def _is_review_text(self, text: str) -> bool:
        """Check if text looks like a review (not navigation/UI text)."""
        if not text or len(text) < 50 or len(text) > 3000:
            return False
        
        review_keywords = [
            'good', 'bad', 'excellent', 'poor', 'quality', 'product',
            'bought', 'purchased', 'using', 'recommend', 'worth',
            'price', 'value', 'satisfied', 'disappointed', 'delivery',
            'camera', 'battery', 'phone', 'screen', 'performance',
            'nice', 'awesome', 'love', 'best', 'worst', 'issue',
            'great', 'amazing', 'perfect', 'terrible', 'waste'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in review_keywords if keyword in text_lower)
        
        return keyword_count >= 2

    def _infer_rating_from_text(self, text: str) -> float:
        """Infer rating from sentiment in text."""
        text_lower = text.lower()
        
        positive_words = ['excellent', 'amazing', 'great', 'good', 'love', 'perfect', 'awesome', 'best']
        negative_words = ['bad', 'poor', 'worst', 'terrible', 'awful', 'hate', 'disappointed', 'waste']
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count > neg_count:
            return 4.5 if pos_count >= 3 else 4.0
        elif neg_count > pos_count:
            return 1.5 if neg_count >= 3 else 2.0
        
        return 3.0

    def _scrape_amazon_with_pagination(self, url: str, max_pages: int = 8) -> List[Dict]:
        """Amazon pagination support."""
        all_reviews = []
        
        if '/product-reviews/' in url:
            base_url = url.split('?')[0]
        else:
            asin_match = re.search(r'/dp/([A-Z0-9]{10})', url)
            if not asin_match:
                return []
            domain = urlparse(url).netloc
            base_url = f"https://{domain}/product-reviews/{asin_match.group(1)}"
        
        for page in range(1, max_pages + 1):
            try:
                page_url = base_url if page == 1 else f"{base_url}?pageNumber={page}"
                print(f"[AMAZON] Fetching page {page}")
                
                response = self.session.get(page_url, headers=self.headers, timeout=15)
                if response.status_code != 200:
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                page_reviews = self._scrape_amazon_reviews(soup)
                
                if not page_reviews:
                    break
                
                all_reviews.extend(page_reviews)
                print(f"[AMAZON] Page {page}: Found {len(page_reviews)} reviews")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"[AMAZON] Error on page {page}: {e}")
                break
        
        print(f"[AMAZON] ✅ Total extracted: {len(all_reviews)} reviews")
        return all_reviews

    def _scrape_amazon_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse Amazon reviews from page."""
        reviews = []
        containers = soup.find_all('div', {'data-hook': 'review'})
        
        for container in containers:
            try:
                rating = 3.0
                rating_elem = container.find('i', {'data-hook': 'review-star-rating'})
                if rating_elem:
                    match = re.search(r'(\d+\.?\d*)', rating_elem.get_text())
                    if match:
                        rating = float(match.group(1))
                
                body_elem = container.find('span', {'data-hook': 'review-body'})
                body = body_elem.get_text(strip=True) if body_elem else ""
                
                title_elem = container.find('a', {'data-hook': 'review-title'})
                title = title_elem.get_text(strip=True) if title_elem else ""
                
                author_elem = container.find('span', class_='a-profile-name')
                author = author_elem.get_text(strip=True) if author_elem else "Anonymous"
                
                date_elem = container.find('span', {'data-hook': 'review-date'})
                date = date_elem.get_text(strip=True) if date_elem else ""
                
                if body and len(body) > 30:
                    reviews.append({
                        'rating': rating,
                        'title': title,
                        'body': body,
                        'author': author,
                        'date': date,
                        'helpful': ""
                    })
                    
            except Exception:
                continue
        
        return reviews

    def _scrape_myntra_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Scrape Myntra reviews."""
        reviews = []
        containers = soup.find_all('div', class_=re.compile(r'user-review', re.I))
        
        for container in containers:
            try:
                text = container.get_text(strip=True)
                if len(text) > 30 and self._is_review_text(text):
                    reviews.append({
                        'rating': 3.0,
                        'title': "",
                        'body': text,
                        'author': "Anonymous",
                        'date': "",
                        'helpful': ""
                    })
            except:
                continue
        
        return reviews

    def _scrape_generic_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Generic review scraping for unknown platforms."""
        reviews = []
        
        for pattern in [r'review', r'comment', r'feedback']:
            containers = soup.find_all('div', class_=re.compile(pattern, re.I))
            
            for container in containers[:30]:
                try:
                    text = container.get_text(strip=True)
                    if 50 < len(text) < 1000 and self._is_review_text(text):
                        reviews.append({
                            'rating': 3.0,
                            'title': "",
                            'body': text,
                            'author': "Anonymous",
                            'date': "",
                            'helpful': ""
                        })
                except:
                    continue
            
            if reviews:
                break
        
        return reviews

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        """Remove scripts, styles, and navigation elements."""
        for tag in ['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']:
            for elem in soup.find_all(tag):
                elem.decompose()

    def _get_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title = soup.find('title')
        if title:
            return title.get_text(strip=True)
        
        h1 = soup.find('h1')
        return h1.get_text(strip=True) if h1 else "Untitled"

    def _get_content(self, soup: BeautifulSoup) -> str:
        """Extract main page content."""
        container = soup.find('body')
        if not container:
            return ""
        
        text_parts = []
        for tag in ['p', 'h1', 'h2', 'h3', 'div', 'span']:
            for elem in container.find_all(tag):
                text = elem.get_text(strip=True)
                if len(text) > 10:
                    text_parts.append(text)
        
        seen = set()
        cleaned = []
        for text in text_parts:
            if len(text) > 20 and text not in seen:
                cleaned.append(text)
                seen.add(text)
        
        return ' '.join(cleaned)

    def _get_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """
        ENHANCED: Extract page metadata including news-specific data.
        """
        metadata = {}
        
        # Description
        desc = soup.find('meta', {'name': 'description'})
        if desc and desc.get('content'):
            metadata['description'] = desc['content']
        
        # OG tags
        og_title = soup.find('meta', {'property': 'og:title'})
        if og_title and og_title.get('content'):
            metadata['og_title'] = og_title['content']
        
        og_type = soup.find('meta', {'property': 'og:type'})
        if og_type and og_type.get('content'):
            metadata['og_type'] = og_type['content']
        
        # NEW: Publication date (for news articles)
        date_selectors = [
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'name': 'pubdate'}),
            ('meta', {'name': 'publishdate'}),
            ('time', {'class': re.compile('date|time|published', re.I)}),
            ('span', {'class': re.compile('date|time|published', re.I)}),
        ]
        
        for tag, attrs in date_selectors:
            elem = soup.find(tag, attrs)
            if elem:
                date = elem.get('content') or elem.get('datetime') or elem.get_text()
                if date:
                    metadata['published_date'] = date.strip()
                    break
        
        # NEW: Author (for news/articles)
        author_selectors = [
            ('meta', {'name': 'author'}),
            ('meta', {'property': 'article:author'}),
            ('span', {'class': re.compile('author|byline', re.I)}),
            ('a', {'rel': 'author'}),
        ]
        
        for tag, attrs in author_selectors:
            elem = soup.find(tag, attrs)
            if elem:
                author = elem.get('content') or elem.get_text()
                if author:
                    metadata['author'] = author.strip()
                    break
        
        # NEW: Keywords/tags
        keywords = soup.find('meta', {'name': 'keywords'})
        if keywords and keywords.get('content'):
            metadata['keywords'] = keywords['content']
        
        return metadata


# Main function
def scrape_webpage(url: str, max_pages: int = 8) -> Dict:
    """
    Main scraping function with enhanced content type detection.
    
    Args:
        url: URL to scrape
        max_pages: Maximum pages to scrape for reviews (default: 8)
    
    Returns:
        Dictionary with scraped data including content_type for summarizer
    """
    scraper = WebpageScraper()
    return scraper.scrape(url, max_pages=max_pages)