"""
Reverse Image Search Module
Finds original sources of images to detect reused/old images
Critical for detecting viral image-based misinformation
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

# API Keys from environment
SERPAPI_KEY = os.getenv('SERPAPI_KEY', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_SEARCH_API_KEY', '')
GOOGLE_CSE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')

class ReverseImageSearch:
    """
    Reverse image search to find original sources
    Uses multiple services: SerpAPI (Google Lens), Yandex, TinEye
    """

    def __init__(self):
        self.serpapi_key = SERPAPI_KEY
        self.google_api_key = GOOGLE_API_KEY
        self.google_cse_id = GOOGLE_CSE_ID

    def generate_image_hash(self, image_data: bytes) -> Dict[str, str]:
        """
        Generate multiple hashes for image comparison
        - MD5: Exact duplicate detection
        - pHash (perceptual hash): Similar image detection
        - aHash (average hash): Fast similarity
        """
        try:
            # MD5 hash for exact duplicates
            md5_hash = hashlib.md5(image_data).hexdigest()

            # Perceptual hashes for similar images
            image = Image.open(BytesIO(image_data))

            # Convert to RGB if necessary
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
        """
        Search using Google Lens via SerpAPI
        Most accurate for finding original sources
        """
        if not self.serpapi_key:
            logger.warning("SerpAPI key not configured")
            return {'success': False, 'error': 'SerpAPI key not configured'}

        try:
            # Save image temporarily as base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # SerpAPI Google Lens endpoint
            params = {
                'engine': 'google_lens',
                'api_key': self.serpapi_key,
                'image': image_base64[:1000000]  # Max 1MB for base64
            }

            logger.info("🔍 Performing Google Lens reverse image search via SerpAPI")

            response = requests.get(
                'https://serpapi.com/search',
                params=params,
                timeout=15
            )

            if response.status_code != 200:
                logger.warning(f"SerpAPI returned status {response.status_code}")
                return {'success': False, 'error': f'API returned {response.status_code}'}

            data = response.json()

            # Parse results
            results = []

            # Visual matches
            visual_matches = data.get('visual_matches', [])
            for match in visual_matches[:5]:  # Top 5
                results.append({
                    'title': match.get('title', ''),
                    'link': match.get('link', ''),
                    'source': match.get('source', ''),
                    'thumbnail': match.get('thumbnail', ''),
                    'type': 'visual_match'
                })

            # Knowledge graph (if famous image)
            knowledge_graph = data.get('knowledge_graph', {})
            if knowledge_graph:
                results.append({
                    'title': knowledge_graph.get('title', ''),
                    'description': knowledge_graph.get('description', ''),
                    'source': knowledge_graph.get('source', ''),
                    'type': 'knowledge_graph'
                })

            logger.info(f"🔍 Found {len(results)} Google Lens results")

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
        """
        Search using Yandex Image Search
        Good for finding Russian/international sources
        """
        try:
            # Yandex uses image upload
            # This is a simplified version - production should use Yandex API
            logger.info("🔍 Yandex image search (placeholder - requires API key)")

            # For now, return placeholder
            # TODO: Implement Yandex Image Search API

            return {
                'success': False,
                'error': 'Yandex API not implemented yet',
                'service': 'yandex'
            }

        except Exception as e:
            logger.error(f"Yandex search failed: {e}")
            return {'success': False, 'error': str(e)}

    def search_tineye(self, image_data: bytes) -> Dict[str, Any]:
        """
        Search using TinEye API
        Excellent for finding oldest occurrence of image
        """
        try:
            # TinEye API requires API key
            # This is a placeholder for future implementation
            logger.info("🔍 TinEye search (placeholder - requires API key)")

            # TODO: Implement TinEye API

            return {
                'success': False,
                'error': 'TinEye API not implemented yet',
                'service': 'tineye'
            }

        except Exception as e:
            logger.error(f"TinEye search failed: {e}")
            return {'success': False, 'error': str(e)}

    def reverse_search_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Comprehensive reverse image search using all available services
        Returns original sources, dates, and context
        """
        try:
            logger.info("🔍 Starting comprehensive reverse image search")

            # Generate image hashes
            image_hashes = self.generate_image_hash(image_data)

            # Search multiple services
            all_results = []

            # 1. Google Lens (primary)
            google_results = self.search_google_lens(image_data)
            if google_results.get('success'):
                all_results.extend(google_results.get('results', []))

            # 2. Yandex (if API available)
            # yandex_results = self.search_yandex_images(image_data)
            # if yandex_results.get('success'):
            #     all_results.extend(yandex_results.get('results', []))

            # 3. TinEye (if API available)
            # tineye_results = self.search_tineye(image_data)
            # if tineye_results.get('success'):
            #     all_results.extend(tineye_results.get('results', []))

            # Analyze results
            analysis = self._analyze_reverse_search_results(all_results, image_hashes)

            return {
                'success': True,
                'image_hashes': image_hashes,
                'results': all_results[:10],  # Top 10 results
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
        """
        Analyze reverse search results to detect image reuse
        """
        analysis = {
            'is_reused_image': False,
            'original_context': None,
            'oldest_source': None,
            'credible_sources': [],
            'red_flags': []
        }

        if not results:
            return analysis

        # Check if image appears on credible news sites
        credible_domains = [
            'thehindu.com', 'indianexpress.com', 'reuters.com', 'pti.in',
            'bbc.com', 'cnn.com', 'apnews.com', 'afp.com'
        ]

        for result in results:
            link = result.get('link', '').lower()

            # Check for credible sources
            for domain in credible_domains:
                if domain in link:
                    analysis['credible_sources'].append({
                        'title': result.get('title'),
                        'link': result.get('link'),
                        'domain': domain
                    })

            # Check for knowledge graph (famous image)
            if result.get('type') == 'knowledge_graph':
                analysis['original_context'] = result.get('description')

        # Red flags
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


# Global instance
_reverse_image_search = None

def get_reverse_image_searcher():
    """Get singleton instance of reverse image search"""
    global _reverse_image_search
    if _reverse_image_search is None:
        _reverse_image_search = ReverseImageSearch()
    return _reverse_image_search
