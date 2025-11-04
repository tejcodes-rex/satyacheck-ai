"""
Reverse Image Search Module
"""

import requests
import hashlib
import imagehash
from PIL import Image
from io import BytesIO
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlencode
import os
from datetime import datetime
import base64

logger = logging.getLogger(__name__)

SERPAPI_KEY = os.getenv('SERPAPI_KEY', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_SEARCH_API_KEY', '')
GOOGLE_CSE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')

class ReverseImageSearch:
    """Reverse image search to find original sources."""

    def __init__(self):
        self.serpapi_key = SERPAPI_KEY
        self.google_api_key = GOOGLE_API_KEY
        self.google_cse_id = GOOGLE_CSE_ID

    def generate_image_hash(self, image_data: bytes) -> Dict[str, str]:
        """Generate multiple hashes for image comparison."""
        try:
            md5_hash = hashlib.md5(image_data).hexdigest()
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            phash = str(imagehash.phash(image))
            ahash = str(imagehash.average_hash(image))
            dhash = str(imagehash.dhash(image))

            return {
                'md5': md5_hash,
                'phash': phash,
                'ahash': ahash,
                'dhash': dhash,
                'dimensions': f"{image.width}x{image.height}",
                'format': image.format or 'unknown'
            }

        except Exception as e:
            logger.error(f"Image hash generation failed: {e}")
            return {
                'md5': hashlib.md5(image_data).hexdigest(),
                'error': str(e)
            }

    def search_google_lens(self, image_data: bytes) -> Dict[str, Any]:
        """Search using Google Lens via SerpAPI."""
        if not self.serpapi_key:
            logger.warning("SerpAPI key not configured")
            return {'success': False, 'error': 'SerpAPI key not configured'}

        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            params = {
                'engine': 'google_lens',
                'api_key': self.serpapi_key,
                'image': image_base64[:1000000]
            }

            logger.info("Performing Google Lens reverse image search via SerpAPI")

            response = requests.get(
                'https://serpapi.com/search',
                params=params,
                timeout=15
            )

            if response.status_code != 200:
                logger.warning(f"SerpAPI returned status {response.status_code}")
                return {'success': False, 'error': f'API returned {response.status_code}'}

            data = response.json()

            results = []
            visual_matches = data.get('visual_matches', [])
            for match in visual_matches[:5]:
                results.append({
                    'title': match.get('title', ''),
                    'link': match.get('link', ''),
                    'source': match.get('source', ''),
                    'thumbnail': match.get('thumbnail', ''),
                    'type': 'visual_match'
                })

            knowledge_graph = data.get('knowledge_graph', {})
            if knowledge_graph:
                results.append({
                    'title': knowledge_graph.get('title', ''),
                    'description': knowledge_graph.get('description', ''),
                    'source': knowledge_graph.get('source', ''),
                    'type': 'knowledge_graph'
                })

            logger.info(f"Found {len(results)} Google Lens results")

            return {
                'success': True,
                'results': results,
                'total_results': len(results),
                'service': 'google_lens'
            }

        except Exception as e:
            logger.error(f"Google Lens search failed: {e}")
            return {'success': False, 'error': str(e)}

    def search_yandex_images(self, image_data: bytes) -> Dict[str, Any]:
        """Search using Yandex Image Search."""
        try:
            logger.info("Yandex image search (placeholder - requires API key)")
            return {
                'success': False,
                'error': 'Yandex API not implemented yet',
                'service': 'yandex'
            }

        except Exception as e:
            logger.error(f"Yandex search failed: {e}")
            return {'success': False, 'error': str(e)}

    def search_tineye(self, image_data: bytes) -> Dict[str, Any]:
        """Search using TinEye API."""
        try:
            logger.info("TinEye search (placeholder - requires API key)")
            return {
                'success': False,
                'error': 'TinEye API not implemented yet',
                'service': 'tineye'
            }

        except Exception as e:
            logger.error(f"TinEye search failed: {e}")
            return {'success': False, 'error': str(e)}

    def reverse_search_image(self, image_data: bytes) -> Dict[str, Any]:
        """Comprehensive reverse image search using all available services."""
        try:
            logger.info("Starting comprehensive reverse image search")

            image_hashes = self.generate_image_hash(image_data)
            all_results = []
            google_results = self.search_google_lens(image_data)
            if google_results.get('success'):
                all_results.extend(google_results.get('results', []))
            analysis = self._analyze_reverse_search_results(all_results, image_hashes)

            return {
                'success': True,
                'image_hashes': image_hashes,
                'results': all_results[:10],
                'total_found': len(all_results),
                'analysis': analysis
            }

        except Exception as e:
            logger.error(f"Reverse image search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_hashes': self.generate_image_hash(image_data)
            }

    def _analyze_reverse_search_results(self, results: List[Dict], image_hashes: Dict) -> Dict[str, Any]:
        """Analyze reverse search results to detect image reuse."""
        analysis = {
            'is_reused_image': False,
            'original_context': None,
            'oldest_source': None,
            'credible_sources': [],
            'red_flags': []
        }

        if not results:
            return analysis

        credible_domains = [
            'thehindu.com', 'indianexpress.com', 'reuters.com', 'pti.in',
            'bbc.com', 'cnn.com', 'apnews.com', 'afp.com'
        ]

        for result in results:
            link = result.get('link', '').lower()

            for domain in credible_domains:
                if domain in link:
                    analysis['credible_sources'].append({
                        'title': result.get('title'),
                        'link': result.get('link'),
                        'domain': domain
                    })

            if result.get('type') == 'knowledge_graph':
                analysis['original_context'] = result.get('description')

        if len(results) > 5:
            analysis['is_reused_image'] = True
            analysis['red_flags'].append(
                f"Image found on {len(results)} different sources - likely reused/old image"
            )

        if analysis['credible_sources']:
            analysis['red_flags'].append(
                f"Image previously published by {len(analysis['credible_sources'])} credible news organizations"
            )

        if analysis['original_context']:
            analysis['red_flags'].append(
                "Image is well-known and has established context"
            )

        return analysis


_reverse_image_search = None

def get_reverse_image_searcher():
    """Get singleton instance of reverse image search."""
    global _reverse_image_search
    if _reverse_image_search is None:
        _reverse_image_search = ReverseImageSearch()
    return _reverse_image_search
