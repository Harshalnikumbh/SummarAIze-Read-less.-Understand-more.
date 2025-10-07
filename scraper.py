import requests
from bs4 import BeautifulSoup
from typing import Dict, List


class WebpageScraper:
    """
    Clean webpage scraper for extracting text content from any URL.
    Designed to feed content to BART summarization model.
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
    
    def scrape(self, url: str) -> Dict:
        """
        Scrape webpage content.
        
        Args:
            url: The webpage URL to scrape
            
        Returns:
            Dictionary containing title, content, and metadata
        """
        try:
            print(f"\n[SCRAPER] Fetching URL: {url}")
            response = self.session.get(url, headers=self.headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            print(f"[SCRAPER] Response status: {response.status_code}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            self._remove_noise(soup)
            
            title = self._get_title(soup)
            content = self._get_content(soup)
            metadata = self._get_metadata(soup)
            
            print(f"[SCRAPER] Title: {title}")
            print(f"[SCRAPER] Content length: {len(content)} characters")
            
            return {
                'success': True,
                'title': title,
                'content': content,
                'metadata': metadata,
                'url': url
            }
            
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
    
    def _remove_noise(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements like scripts, ads, navigation."""
        unwanted = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']
        
        for tag in unwanted:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements with ad-related classes (but be less aggressive)
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
        
        # Extract text from common tags - LESS STRICT filtering
        text_parts = []
        text_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div']
        
        for tag in text_tags:
            for element in container.find_all(tag):
                text = element.get_text(separator=' ', strip=True)
                # REDUCED minimum length from 30 to 10
                if len(text) > 10:
                    text_parts.append(text)
        
        print(f"[SCRAPER] Found {len(text_parts)} text parts")
        
        # If still no content, get ALL text from container
        if len(text_parts) == 0:
            print("[SCRAPER] No text parts found, getting all text from container")
            all_text = container.get_text(separator=' ', strip=True)
            if all_text:
                text_parts = [all_text]
        
        # Clean and deduplicate - LESS AGGRESSIVE
        cleaned = []
        seen = set()
        
        for text in text_parts:
            # Clean whitespace
            text = ' '.join(text.split())
            
            # REDUCED minimum length from 30 to 20
            # Allow some duplicates for better content coverage
            if len(text) > 20:
                # Only skip if EXACT duplicate
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


def scrape_webpage(url: str) -> Dict:
    """
    Scrape any webpage and extract text content.
    
    Args:
        url: Webpage URL to scrape
        
    Returns:
        Dictionary with scraped content ready for BART
    """
    scraper = WebpageScraper()
    return scraper.scrape(url)


if __name__ == "__main__":
    # Test the scraper
    test_url = "https://example.com"
    result = scrape_webpage(test_url)
    
    if result['success']:
        print(f"\n✓ Title: {result['title']}")
        print(f"✓ Content length: {len(result['content'])} characters")
        print(f"\nFirst 500 characters:\n{result['content'][:500]}...")
    else:
        print(f"\n✗ Error: {result['error']}")