import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re
import time
from urllib.parse import urlparse, parse_qs


class WebpageScraper:
    """
    Enhanced webpage scraper with FIXED Flipkart review detection.
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
        """
        try:
            print(f"\n[SCRAPER] Fetching URL: {url}")
            response = self.session.get(url, headers=self.headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            print(f"[SCRAPER] Response status: {response.status_code}")

            soup = BeautifulSoup(response.content, 'html.parser')

            # Enhanced detection logic
            url_type = self._detect_url_type(url, soup)
            print(f"[SCRAPER] URL Type: {url_type}")

            self._remove_noise(soup)

            title = self._get_title(soup)
            content = self._get_content(soup)
            metadata = self._get_metadata(soup)

            result = {
                'success': True,
                'title': title,
                'content': content,
                'metadata': metadata,
                'url': url,
                'url_type': url_type,
                'reviews': []
            }

            # Scrape reviews only if it's a review page or product page
            if url_type in ['review_page', 'product_page']:
                reviews = self._scrape_reviews_with_pagination(soup, url, max_pages)
                result['reviews'] = reviews
                print(f"[SCRAPER] Extracted {len(reviews)} total reviews")

            print(f"[SCRAPER] Title: {title}")
            print(f"[SCRAPER] Content length: {len(content)} characters")

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

    def _detect_url_type(self, url: str, soup: BeautifulSoup) -> str:
        """
        Detect if URL is a review page, product page, or regular webpage.
        """
        url_lower = url.lower()

        # Check for explicit review page patterns
        review_page_patterns = [
            r'/product-reviews/',
            r'/customer-reviews/',
            r'/reviews/',
        ]

        for pattern in review_page_patterns:
            if re.search(pattern, url_lower):
                print(f"[DETECTION] Review page detected by URL pattern: {pattern}")
                return 'review_page'

        # Check for product page patterns
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

        # Check for product schema
        product_schema = soup.find(attrs={'itemtype': re.compile(r'Product', re.IGNORECASE)})
        if product_schema:
            print(f"[DETECTION] Product page detected by schema")
            return 'product_page'

        print(f"[DETECTION] Regular webpage detected")
        return 'webpage'

    def _scrape_reviews_with_pagination(self, soup: BeautifulSoup, url: str, max_pages: int = 8) -> List[Dict]:
        """Scrape product reviews with enhanced pagination support."""
        all_reviews = []

        # Determine platform
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
        """Simplified: Use HTML fallback for Flipkart since API blocks external requests."""
        print("[FLIPKART] Using HTML mode (API blocked with 403).")
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
                time.sleep(1)
            except Exception as e:
                print(f"[FLIPKART] Error on page {page}: {e}")
                break
        print(f"[FLIPKART] ✅ Extracted {len(all_reviews)} total reviews (HTML mode)")
        return all_reviews

    def _scrape_flipkart_reviews_from_page(self, url: str) -> List[Dict]:
        """Scrape reviews from a single Flipkart page."""
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            return self._parse_flipkart_reviews(soup)
        except Exception as e:
            print(f"[FLIPKART] Error scraping page: {e}")
            return []

    def _parse_flipkart_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """COMPLETELY UPDATED Flipkart review parser with NEW selectors."""
        reviews = []
        print(f"[FLIPKART] Parsing reviews with updated selectors...")

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
                    print(f"[FLIPKART] Strategy found {len(containers)} containers")
                    if len(all_containers) > 50:
                        break
            except Exception:
                continue

        print(f"[FLIPKART] Total containers to check: {len(all_containers)}")

# DEBUG: Print first 3 container texts
        for i, container in enumerate(all_containers[:3]):
            text = container.get_text(strip=True)[:200]
            print(f"[DEBUG] Container {i+1}: {text}")

        seen_texts = set()
        for container in all_containers:
            try:
                full_text = container.get_text(separator=' ', strip=True)
                if len(full_text) < 50 or full_text in seen_texts:
                    continue
                if not self._is_review_text(full_text):
                    continue
                seen_texts.add(full_text)

                review_data = self._extract_flipkart_review_smart(container)
                if review_data and review_data.get('body') and len(review_data['body']) > 30:
                    reviews.append(review_data)
                if len(reviews) >= 20:
                    break
            except Exception:
                continue

        print(f"[FLIPKART] Extracted {len(reviews)} valid reviews")
        return reviews

    def _extract_flipkart_review_smart(self, container) -> Optional[Dict]:
        """SMART extraction that works with ANY Flipkart structure."""
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
                author = match.group(1).strip()
                break

        date = ""
        date_match = re.search(r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})', full_text, re.I)
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

    # --- remaining helper methods (unchanged) ---
    def _extract_review_body(self, full_text: str) -> str:
        noise_patterns = [
            r'READ MORE',
            r'Certified Buyer',
            r'\d+\s+people found this helpful',
            r'Was this review helpful\?',
            r'👍\s*\d+',
            r'👎\s*\d+',
        ]
        cleaned = full_text
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        sentences = re.split(r'[.!?]+', cleaned)
        valid_sentences = [s.strip() for s in sentences if 30 < len(s.strip()) < 500 and self._is_review_sentence(s)]
        if valid_sentences:
            return ". ".join(valid_sentences[:5]) + "."
        if len(cleaned) > 30:
            return cleaned[:500].strip()
        return ""

    def _is_review_sentence(self, text: str) -> bool:
        text_lower = text.lower()
        review_words = [
            'product', 'quality', 'good', 'bad', 'excellent', 'poor',
            'buy', 'bought', 'purchase', 'order', 'delivery', 'price',
            'use', 'using', 'used', 'work', 'works', 'phone', 'camera',
            'battery', 'screen', 'display', 'performance', 'recommend',
            'satisfied', 'disappointed', 'love', 'like', 'hate', 'worth'
        ]
        return any(word in text_lower for word in review_words)

    def _is_review_text(self, text: str) -> bool:
        if not text or len(text) < 50 or len(text) > 3000:
            return False
        review_keywords = [
            'good', 'bad', 'excellent', 'poor', 'quality', 'product',
            'bought', 'purchased', 'using', 'recommend', 'worth',
            'price', 'value', 'satisfied', 'disappointed', 'delivery',
            'camera', 'battery', 'phone', 'screen', 'performance',
            'nice', 'awesome', 'love', 'best', 'worst', 'issue'
        ]
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in review_keywords if keyword in text_lower)
        # Changed from >= 3 to >= 2 (more lenient)
        return keyword_count >= 2

    # --- Amazon, Myntra, and generic scrapers unchanged ---
    def _scrape_amazon_with_pagination(self, url: str, max_pages: int = 8) -> List[Dict]:
        """Amazon pagination."""
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
                response = self.session.get(page_url, headers=self.headers, timeout=15)
                if response.status_code != 200:
                    break
                soup = BeautifulSoup(response.content, 'html.parser')
                page_reviews = self._scrape_amazon_reviews(soup)
                if not page_reviews:
                    break
                all_reviews.extend(page_reviews)
                time.sleep(1)
            except:
                break
        return all_reviews

    def _scrape_amazon_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse Amazon reviews."""
        reviews = []
        containers = soup.find_all('div', {'data-hook': 'review'})
        for container in containers:
            try:
                rating_elem = container.find('i', {'data-hook': 'review-star-rating'})
                rating = 3.0
                if rating_elem:
                    match = re.search(r'(\d+\.?\d*)', rating_elem.get_text())
                    if match:
                        rating = float(match.group(1))
                
                body_elem = container.find('span', {'data-hook': 'review-body'})
                body = body_elem.get_text(strip=True) if body_elem else ""
                
                if body and len(body) > 30:
                    reviews.append({
                        'rating': rating,
                        'title': "",
                        'body': body,
                        'author': "Anonymous",
                        'date': "",
                        'helpful': ""
                    })
            except:
                continue
        return reviews

    def _scrape_myntra_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Myntra reviews."""
        reviews = []
        containers = soup.find_all('div', class_=re.compile(r'user-review'))
        for container in containers:
            try:
                text = container.get_text(strip=True)
                if len(text) > 30:
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
        """Generic review scraping."""
        reviews = []
        for pattern in [r'review', r'comment']:
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

    def _infer_rating_from_text(self, text: str) -> float:
        """Infer rating from sentiment."""
        text_lower = text.lower()
        positive = ['excellent', 'amazing', 'great', 'good', 'love', 'perfect']
        negative = ['bad', 'poor', 'worst', 'terrible', 'awful', 'hate']
        pos = sum(1 for w in positive if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)
        if pos > neg:
            return 4.5 if pos >= 3 else 4.0
        elif neg > pos:
            return 1.5 if neg >= 3 else 2.0
        return 3.0

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        """Remove scripts, styles, etc."""
        for tag in ['script', 'style', 'nav', 'footer', 'header']:
            for elem in soup.find_all(tag):
                elem.decompose()

    def _get_title(self, soup: BeautifulSoup) -> str:
        """Get page title."""
        title = soup.find('title')
        if title:
            return title.get_text(strip=True)
        h1 = soup.find('h1')
        return h1.get_text(strip=True) if h1 else "Untitled"

    def _get_content(self, soup: BeautifulSoup) -> str:
        """Get page content."""
        container = soup.find('body')
        if not container:
            return ""
        text_parts = []
        for tag in ['p', 'h1', 'h2', 'h3', 'div']:
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

    def _get_metadata(self, soup: BeautifulSoup) -> Dict:
        """Get metadata."""
        metadata = {}
        desc = soup.find('meta', {'name': 'description'})
        if desc and desc.get('content'):
            metadata['description'] = desc['content']
        return metadata


def scrape_webpage(url: str, max_pages: int = 8) -> Dict:
    """Main scraping function."""
    scraper = WebpageScraper()
    return scraper.scrape(url, max_pages=max_pages)
