import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re
import time

class WebpageScraper:
    """
    Enhanced webpage scraper for extracting text content and product reviews.
    Supports general webpages and e-commerce product pages.
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
    
    def scrape(self, url: str, scrape_reviews: bool = False) -> Dict:
        """
        Scrape webpage content and optionally product reviews.
        
        Args:
            url: The webpage URL to scrape
            scrape_reviews: Whether to attempt scraping product reviews
            
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
                reviews = self._scrape_reviews(soup, url)
                result['reviews'] = reviews
                print(f"[SCRAPER] Extracted {len(reviews)} reviews")
            
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
            r'/product/',
            r'/dp/',
            r'/item/',
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
    
    def _scrape_reviews(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """
        Scrape product reviews from e-commerce websites.
        Supports Amazon, Flipkart, and other major platforms.
        """
        reviews = []
        
        # Determine platform and use appropriate scraping strategy
        if 'amazon' in url.lower():
            reviews = self._scrape_amazon_reviews(soup)
        elif 'flipkart' in url.lower():
            reviews = self._scrape_flipkart_reviews(soup)
        else:
            # Generic review scraping
            reviews = self._scrape_generic_reviews(soup)
        
        return reviews
    
    def _scrape_amazon_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Scrape reviews from Amazon product pages."""
        reviews = []
        
        # Amazon review selectors
        review_containers = soup.find_all('div', {'data-hook': 'review'})
        
        if not review_containers:
            # Alternative selectors
            review_containers = soup.find_all('div', class_=re.compile(r'review', re.IGNORECASE))
        
        print(f"[AMAZON] Found {len(review_containers)} review containers")
        
        for container in review_containers[:50]:  # Limit to 50 reviews
            try:
                # Extract rating
                rating_elem = container.find('i', {'data-hook': 'review-star-rating'})
                if not rating_elem:
                    rating_elem = container.find('span', class_=re.compile(r'a-icon-star'))
                
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
    
    def _scrape_flipkart_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Scrape reviews from Flipkart product pages."""
        reviews = []
        
        # Flipkart review selectors
        review_containers = soup.find_all('div', class_=re.compile(r'_1AtVbE|col-12-12', re.IGNORECASE))
        
        print(f"[FLIPKART] Found {len(review_containers)} potential review containers")
        
        for container in review_containers[:50]:
            try:
                # Extract rating
                rating_elem = container.find('div', class_=re.compile(r'_3LWZlK|hGSR34'))
                rating = None
                if rating_elem:
                    rating_text = rating_elem.get_text(strip=True)
                    rating_match = re.search(r'(\d+)', rating_text)
                    if rating_match:
                        rating = float(rating_match.group(1))
                
                # Extract review text
                body_elem = container.find('div', class_=re.compile(r't-ZTKy'))
                if not body_elem:
                    body_elem = container.find('div', class_='_2-N8zT')
                body = body_elem.get_text(strip=True) if body_elem else ""
                
                # Extract reviewer name
                author_elem = container.find('p', class_=re.compile(r'_2sc7ZR|_2NsDsF'))
                author = author_elem.get_text(strip=True) if author_elem else "Anonymous"
                
                if body and len(body) > 20:  # Only add substantial reviews
                    reviews.append({
                        'rating': rating,
                        'title': "",
                        'body': body,
                        'author': author,
                        'date': "",
                        'helpful': ""
                    })
            
            except Exception as e:
                print(f"[FLIPKART] Error parsing review: {e}")
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
                    
                    if text and len(text) > 30:
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
    test_url = "https://www.amazon.in/Dopamine-Detox-Remove-Distractions-Productivity-ebook/dp/B098MHBF23/ref=books_storefront_desktop_mfs_ts_1?_encoding=UTF8&pd_rd_w=S92WH&content-id=amzn1.sym.87f92a02-d841-4c88-a23e-65534f93faa3&pf_rd_p=87f92a02-d841-4c88-a23e-65534f93faa3&pf_rd_r=NQ6ZBNSCFN5JTRM9N536&pd_rd_wg=uH4yH&pd_rd_r=3e70fd1c-9b3c-40e8-815a-080c1ffc84d5"  
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