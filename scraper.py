import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
import re
import time
import json
from urllib.parse import urlparse, parse_qs


class WebpageScraper:
    """
    Enhanced webpage scraper for extracting text content and product reviews.
    Supports general webpages and e-commerce product pages with pagination.
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        self.session = requests.Session()
    
    def scrape(self, url: str, scrape_reviews: bool = False, max_pages: int = 3) -> Dict:
        """
        Scrape webpage content and optionally product reviews with pagination.
        
        Args:
            url: The webpage URL to scrape
            scrape_reviews: Whether to attempt scraping product reviews
            max_pages: Maximum number of review pages to scrape for pagination
            
        Returns:
            Dictionary containing title, content, reviews, and metadata
        """
        try:
            print(f"\n[SCRAPER] Fetching URL: {url}")
            response = self.session.get(url, headers=self.headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            print(f"[SCRAPER] Response status: {response.status_code}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Detect if this is an e-commerce product page
            is_product_page = self._is_product_page(url, soup)
            print(f"[SCRAPER] Product page detected: {is_product_page}")
            
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
                'is_product_page': is_product_page,
                'reviews': []
            }
            
            # Scrape reviews if requested and this is a product page
            if scrape_reviews or is_product_page:
                reviews = self._scrape_reviews_with_pagination(soup, url, max_pages)
                result['reviews'] = reviews
                print(f"[SCRAPER] Extracted {len(reviews)} total reviews")
            
            print(f"[SCRAPER] Title: {title}")
            print(f"[SCRAPER] Content length: {len(content)} characters")
            
            return result
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                return {
                    'success': False,
                    'error': 'Access forbidden - Website blocking automated requests.',
                    'status_code': 403,
                    'url': url
                }
            return {
                'success': False,
                'error': f'HTTP Error {e.response.status_code}: {str(e)}',
                'status_code': e.response.status_code,
                'url': url
            }
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timed out',
                'url': url
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': 'Connection failed - Check internet or URL',
                'url': url
            }
        except Exception as e:
            print(f"[SCRAPER] Error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _is_product_page(self, url: str, soup: BeautifulSoup) -> bool:
        """Detect if the page is an e-commerce product page."""
        # Check URL patterns
        product_patterns = [
            r'amazon\.(com|in|co\.uk)',
            r'flipkart\.com',
            r'ebay\.(com|in)',
            r'myntra\.com',
            r'ajio\.com',
            r'walmart\.com',
            r'nykaa\.com',
            r'meesho\.com',
            r'/product/',
            r'/dp/',
            r'/item/',
            r'/p/',
        ]
        
        for pattern in product_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        # Check for common product page elements
        product_indicators = [
            soup.find(id=re.compile(r'product|item', re.IGNORECASE)),
            soup.find(class_=re.compile(r'product|item', re.IGNORECASE)),
            soup.find('div', {'data-component-type': 'product'}),
            soup.find(attrs={'itemtype': re.compile(r'Product', re.IGNORECASE)})
        ]
        
        return any(indicator for indicator in product_indicators)
    
    def _scrape_reviews_with_pagination(self, soup: BeautifulSoup, url: str, max_pages: int = 3) -> List[Dict]:
        """
        Scrape product reviews with pagination support.
        """
        all_reviews = []
        
        # Determine platform
        if 'flipkart' in url.lower():
            all_reviews = self._scrape_flipkart_with_pagination(url, max_pages)
        elif 'amazon' in url.lower():
            all_reviews = self._scrape_amazon_reviews(soup)
        elif 'myntra' in url.lower():
            all_reviews = self._scrape_myntra_reviews(soup)
        else:
            # Generic review scraping
            all_reviews = self._scrape_generic_reviews(soup)
        
        return all_reviews
    
    def _scrape_flipkart_with_pagination(self, url: str, max_pages: int = 3) -> List[Dict]:
        """
        Scrape Flipkart reviews with pagination support.
        """
        all_reviews = []
        
        # Check if this is already a reviews page or product page
        if '/product-reviews/' in url:
            base_url = url.split('?')[0]
        else:
            # Try to construct reviews URL from product URL
            # Flipkart pattern: /product-name/product-reviews/itm...
            if '/p/itm' in url:
                product_id = re.search(r'/p/(itm[a-zA-Z0-9]+)', url)
                if product_id:
                    # Extract base URL and construct reviews URL
                    parts = url.split('/p/')
                    base_url = parts[0] + '/product-reviews/' + product_id.group(1)
                else:
                    # Fallback to scraping current page only
                    return self._scrape_flipkart_reviews_from_page(url)
            else:
                return self._scrape_flipkart_reviews_from_page(url)
        
        print(f"[FLIPKART] Base reviews URL: {base_url}")
        
        # Scrape multiple pages
        for page_num in range(1, max_pages + 1):
            try:
                # Construct page URL
                if page_num == 1:
                    page_url = base_url
                else:
                    page_url = f"{base_url}?page={page_num}"
                
                print(f"[FLIPKART] Scraping page {page_num}: {page_url}")
                
                response = self.session.get(page_url, headers=self.headers, timeout=15)
                if response.status_code != 200:
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                page_reviews = self._parse_flipkart_reviews(soup)
                
                if not page_reviews:
                    print(f"[FLIPKART] No reviews found on page {page_num}, stopping pagination")
                    break
                
                all_reviews.extend(page_reviews)
                print(f"[FLIPKART] Found {len(page_reviews)} reviews on page {page_num}")
                
                # Small delay to avoid rate limiting
                if page_num < max_pages:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"[FLIPKART] Error on page {page_num}: {e}")
                break
        
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
        """Parse Flipkart reviews from a page with improved selectors."""
        reviews = []
        
        # Enhanced selectors for Flipkart reviews
        review_selectors = [
            # Main review containers
            ('div', {'class': re.compile(r'_27M-vq|_1AtVbE.*col-12-12|_2vzgPQ')}),
            ('div', {'class': re.compile(r'_1PBCrt|_3nrCtb')}),
            # Individual review cards
            ('div', {'class': re.compile(r'col _2wzgFH|_1U-xdw')}),
            # Review sections
            ('div', {'class': 'row'}),
        ]
        
        all_containers = []
        for tag, attrs in review_selectors:
            containers = soup.find_all(tag, attrs)
            if containers:
                all_containers.extend(containers)
                
        print(f"[FLIPKART] Found {len(all_containers)} potential review containers")
        
        for container in all_containers[:100]:
            try:
                review_data = self._extract_flipkart_review(container)
                if review_data and review_data.get('body'):
                    reviews.append(review_data)
                    
            except Exception as e:
                continue
        
        return reviews
    
    def _extract_flipkart_review(self, container) -> Optional[Dict]:
        """Extract review data from a Flipkart review container."""
        # Extract rating
        rating = None
        
        # Multiple methods to find rating
        rating_selectors = [
            ('div', {'class': re.compile(r'_3LWZlK|_1BLPMq|hGSR34')}),
            ('div', {'class': re.compile(r'_2d4LTz')}),
            ('span', {'class': re.compile(r'_2_R_DZ')}),
        ]
        
        for tag, attrs in rating_selectors:
            rating_elem = container.find(tag, attrs)
            if rating_elem:
                rating_text = rating_elem.get_text(strip=True)
                # Extract number from text like "5★" or "5"
                rating_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                if rating_match:
                    rating = float(rating_match.group(1))
                    break
        
        # If still no rating, look for star icons
        if not rating:
            # Count filled stars (usually have specific class)
            filled_stars = container.find_all('svg', {'class': re.compile(r'_1wB99o|_3LWZlK')})
            if filled_stars:
                rating = len(filled_stars)
        
        # Extract review text
        body = ""
        text_selectors = [
            ('div', {'class': re.compile(r't-ZTKy|_6K-7Co')}),
            ('div', {'class': 'qwjRop'}),
            ('div', {'class': re.compile(r'_2-N8zT')}),
            ('p', {}),  # Generic paragraph
            ('span', {'class': re.compile(r'_2-N8zT')}),
        ]
        
        for tag, attrs in text_selectors:
            if attrs:
                body_elem = container.find(tag, attrs)
            else:
                body_elem = container.find(tag)
            
            if body_elem:
                body = body_elem.get_text(strip=True)
                if len(body) > 30:  # Valid review text
                    break
        
        # If still no body, try getting all text
        if not body or len(body) < 30:
            full_text = container.get_text(separator=' ', strip=True)
            # Check if it looks like a review
            if self._is_review_text(full_text):
                body = full_text
        
        # Extract title
        title = ""
        title_selectors = [
            ('p', {'class': '_2-N8zT'}),
            ('div', {'class': '_2t8wE0'}),
        ]
        
        for tag, attrs in title_selectors:
            title_elem = container.find(tag, attrs)
            if title_elem:
                title = title_elem.get_text(strip=True)
                break
        
        # Extract reviewer name
        author = "Anonymous"
        author_selectors = [
            ('p', {'class': '_2sc7ZR'}),
            ('span', {'class': '_2V5EHH'}),
            ('div', {'class': '_2NsDsF'}),
        ]
        
        for tag, attrs in author_selectors:
            author_elem = container.find(tag, attrs)
            if author_elem:
                author_text = author_elem.get_text(strip=True)
                # Clean up author name (remove "Certified Buyer" etc)
                author = author_text.split(',')[0].strip()
                break
        
        # Extract date
        date = ""
        date_elem = container.find(text=re.compile(r'\d+\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'))
        if date_elem:
            date = date_elem.strip()
        
        # Validate and return
        if body and len(body) > 30:
            # If no rating found, try to infer from sentiment
            if not rating:
                rating = self._infer_rating_from_text(body)
            
            return {
                'rating': rating,
                'title': title,
                'body': body,
                'author': author,
                'date': date,
                'helpful': ""
            }
        
        return None
    
    def _is_review_text(self, text: str) -> bool:
        """Check if text looks like a review."""
        if not text or len(text) < 50 or len(text) > 3000:
            return False
        
        review_keywords = [
            'good', 'bad', 'excellent', 'poor', 'quality', 'product',
            'bought', 'purchased', 'using', 'recommend', 'worth',
            'price', 'value', 'satisfied', 'disappointed', 'love',
            'hate', 'awesome', 'terrible', 'amazing', 'worst'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in review_keywords if keyword in text_lower)
        
        return keyword_count >= 2
    
    def _infer_rating_from_text(self, text: str) -> float:
        """Infer rating from review text sentiment."""
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = [
            'excellent', 'amazing', 'awesome', 'fantastic', 'great',
            'good', 'love', 'perfect', 'best', 'satisfied', 'happy',
            'recommended', 'worth', 'superb', 'wonderful'
        ]
        
        # Negative indicators
        negative_words = [
            'bad', 'poor', 'worst', 'terrible', 'awful', 'hate',
            'disappointed', 'waste', 'useless', 'pathetic', 'horrible',
            'not recommend', 'defective', 'broken', 'fake'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Infer rating based on sentiment
        if positive_count > negative_count:
            if positive_count >= 3:
                return 5.0
            elif positive_count >= 2:
                return 4.0
            else:
                return 3.5
        elif negative_count > positive_count:
            if negative_count >= 3:
                return 1.0
            elif negative_count >= 2:
                return 2.0
            else:
                return 2.5
        else:
            return 3.0  # Neutral
    
    def _scrape_amazon_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Scrape reviews from Amazon product pages."""
        reviews = []
        
        # Amazon review selectors
        review_containers = soup.find_all('div', {'data-hook': 'review'})
        
        if not review_containers:
            # Alternative selectors
            review_containers = soup.find_all('div', class_=re.compile(r'review|a-section\s+review', re.IGNORECASE))
        
        print(f"[AMAZON] Found {len(review_containers)} review containers")
        
        for container in review_containers[:50]:  # Limit to 50 reviews
            try:
                # Extract rating
                rating_elem = container.find('i', {'data-hook': 'review-star-rating'})
                if not rating_elem:
                    rating_elem = container.find('span', class_=re.compile(r'a-icon-alt'))
                
                rating = None
                if rating_elem:
                    rating_text = rating_elem.get_text(strip=True)
                    rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                    if rating_match:
                        rating = float(rating_match.group(1))
                
                # Extract review title
                title_elem = container.find('a', {'data-hook': 'review-title'})
                if not title_elem:
                    title_elem = container.find('span', {'data-hook': 'review-title'})
                title = title_elem.get_text(strip=True) if title_elem else ""
                
                # Extract review body
                body_elem = container.find('span', {'data-hook': 'review-body'})
                if not body_elem:
                    body_elem = container.find('div', class_=re.compile(r'review-text'))
                body = body_elem.get_text(strip=True) if body_elem else ""
                
                # Extract reviewer name
                author_elem = container.find('span', class_='a-profile-name')
                author = author_elem.get_text(strip=True) if author_elem else "Anonymous"
                
                # Extract date
                date_elem = container.find('span', {'data-hook': 'review-date'})
                date = date_elem.get_text(strip=True) if date_elem else ""
                
                # Extract helpful votes
                helpful_elem = container.find('span', {'data-hook': 'helpful-vote-statement'})
                helpful = helpful_elem.get_text(strip=True) if helpful_elem else ""
                
                if body:  # Only add if we have review text
                    if not rating:
                        rating = self._infer_rating_from_text(body)
                    
                    reviews.append({
                        'rating': rating,
                        'title': title,
                        'body': body,
                        'author': author,
                        'date': date,
                        'helpful': helpful
                    })
            
            except Exception as e:
                print(f"[AMAZON] Error parsing review: {e}")
                continue
        
        return reviews
    
    def _scrape_myntra_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Scrape reviews from Myntra product pages."""
        reviews = []
        
        # Myntra review selectors
        review_selectors = [
            ('div', {'class': re.compile(r'user-review-.*|review-comment')}),
            ('div', {'class': 'detailed-reviews-userReviewsContainer'}),
            ('div', {'class': 'user-review'}),
        ]
        
        all_containers = []
        for tag, attrs in review_selectors:
            containers = soup.find_all(tag, attrs)
            if containers:
                all_containers.extend(containers)
        
        print(f"[MYNTRA] Found {len(all_containers)} review containers")
        
        for container in all_containers[:50]:
            try:
                # Extract rating (Myntra uses star count)
                rating = None
                stars = container.find_all('span', class_=re.compile(r'icon-star.*filled'))
                if stars:
                    rating = len(stars)
                
                # Extract review text
                body = ""
                text_elem = container.find('div', class_=re.compile(r'user-review-reviewTextWrapper'))
                if not text_elem:
                    text_elem = container.find('div', class_='user-review-text')
                if text_elem:
                    body = text_elem.get_text(strip=True)
                
                # Extract author
                author = "Anonymous"
                author_elem = container.find('div', class_=re.compile(r'user-review-userName'))
                if author_elem:
                    author = author_elem.get_text(strip=True)
                
                # Extract date
                date = ""
                date_elem = container.find('div', class_=re.compile(r'user-review-date'))
                if date_elem:
                    date = date_elem.get_text(strip=True)
                
                if body and len(body) > 30:
                    if not rating:
                        rating = self._infer_rating_from_text(body)
                    
                    reviews.append({
                        'rating': rating,
                        'title': "",
                        'body': body,
                        'author': author,
                        'date': date,
                        'helpful': ""
                    })
            
            except Exception as e:
                print(f"[MYNTRA] Error parsing review: {e}")
                continue
        
        return reviews
    
    def _scrape_generic_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Generic review scraping for unknown e-commerce platforms."""
        reviews = []
        
        # Common review-related class/id patterns
        review_patterns = [
            r'review',
            r'comment',
            r'feedback',
            r'testimonial',
            r'rating'
        ]
        
        # Find potential review containers
        for pattern in review_patterns:
            containers = soup.find_all(['div', 'article', 'section'], 
                                      class_=re.compile(pattern, re.IGNORECASE))
            
            for container in containers[:30]:
                try:
                    # Extract text content
                    text = container.get_text(separator=' ', strip=True)
                    
                    # Look for rating
                    rating = None
                    rating_elem = container.find(class_=re.compile(r'star|rating', re.IGNORECASE))
                    if rating_elem:
                        rating_text = rating_elem.get_text(strip=True)
                        rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                        if rating_match:
                            rating = float(rating_match.group(1))
                    
                    if not rating:
                        rating = self._infer_rating_from_text(text)
                    
                    if text and len(text) > 30 and self._is_review_text(text):
                        reviews.append({
                            'rating': rating,
                            'title': "",
                            'body': text,
                            'author': "Anonymous",
                            'date': "",
                            'helpful': ""
                        })
                
                except Exception as e:
                    continue
            
            if reviews:  # If we found reviews, stop searching
                break
        
        return reviews
    
    def _remove_noise(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements like scripts, ads, navigation."""
        unwanted = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']
        
        for tag in unwanted:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements with ad-related classes
        for element in soup.find_all(class_=lambda x: x and any(
            noise in str(x).lower() for noise in ['advertisement', 'ad-container', 'ad-banner']
        )):
            element.decompose()
    
    def _get_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title = soup.find('title')
        if title:
            return title.get_text(strip=True)
        
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        
        return "Untitled"
    
    def _get_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from webpage."""
        
        # Find main content container
        containers = [
            soup.find('main'),
            soup.find('article'),
            soup.find('div', class_=lambda x: x and 'content' in str(x).lower()),
            soup.find('div', id=lambda x: x and 'content' in str(x).lower()),
            soup.find('body')
        ]
        
        container = next((c for c in containers if c), None)
        
        if not container:
            print("[SCRAPER] No container found!")
            return ""
        
        print(f"[SCRAPER] Using container: {container.name}")
        
        # Extract text from common tags
        text_parts = []
        text_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div']
        
        for tag in text_tags:
            for element in container.find_all(tag):
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 10:
                    text_parts.append(text)
        
        print(f"[SCRAPER] Found {len(text_parts)} text parts")
        
        # If still no content, get ALL text from container
        if len(text_parts) == 0:
            print("[SCRAPER] No text parts found, getting all text from container")
            all_text = container.get_text(separator=' ', strip=True)
            if all_text:
                text_parts = [all_text]
        
        # Clean and deduplicate
        cleaned = []
        seen = set()
        
        for text in text_parts:
            text = ' '.join(text.split())
            
            if len(text) > 20:
                if text not in seen:
                    cleaned.append(text)
                    seen.add(text)
        
        print(f"[SCRAPER] Cleaned to {len(cleaned)} unique text parts")
        
        final_content = ' '.join(cleaned)
        print(f"[SCRAPER] Final content length: {len(final_content)} characters")
        
        return final_content
    
    def _get_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata like description and keywords."""
        metadata = {}
        
        description = soup.find('meta', attrs={'name': 'description'})
        if not description:
            description = soup.find('meta', attrs={'property': 'og:description'})
        if description and description.get('content'):
            metadata['description'] = description['content']
        
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        if keywords and keywords.get('content'):
            metadata['keywords'] = keywords['content']
        
        return metadata


def scrape_webpage(url: str, scrape_reviews: bool = True) -> Dict:
    """
    Scrape any webpage and extract text content and reviews.
    
    Args:
        url: Webpage URL to scrape
        scrape_reviews: Whether to scrape product reviews (default: True)
        
    Returns:
        Dictionary with scraped content ready for summarization
    """
    scraper = WebpageScraper()
    return scraper.scrape(url, scrape_reviews=scrape_reviews)


if __name__ == "__main__":
    # Test the scraper
    test_urls = [
        "https://www.flipkart.com/example-product/p/itmexample",
        "https://www.amazon.in/dp/B0EXAMPLE",
        "https://www.myntra.com/example-product"
    ]
    
    for test_url in test_urls:
        print(f"\n{'='*60}")
        print(f"Testing: {test_url}")
        print(f"{'='*60}")
        
        result = scrape_webpage(test_url)
        
        if result['success']:
            print(f"\n✓ Title: {result['title']}")
            print(f"✓ Content length: {len(result['content'])} characters")
            print(f"✓ Is product page: {result.get('is_product_page', False)}")
            print(f"✓ Reviews found: {len(result.get('reviews', []))}")
            
            if result.get('reviews'):
                print(f"\nSample review:")
                print(f"Rating: {result['reviews'][0].get('rating')}")
                print(f"Body: {result['reviews'][0].get('body')[:200]}...")
            else:
                print(f"\n✗ Error: {result['error']}")