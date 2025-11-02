"""
SatyaCheck AI - Firestore Database Layer
Production-ready persistent storage with Google Cloud Firestore
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from google.cloud import firestore
from google.api_core import exceptions as google_exceptions

from utils import get_current_timestamp, SimpleCache

logger = logging.getLogger(__name__)

# === DATA MODELS ===

@dataclass
class ClaimRecord:
    """Claim record for persistent storage"""
    claim_id: str
    claim_text: str
    claim_hash: str
    language: str
    verdict: str
    confidence_score: int
    analysis_result: Dict
    created_at: str
    processing_time: float


@dataclass
class HealthStatus:
    """System health status"""
    status: str
    timestamp: str
    services: Dict[str, str]


# === FIRESTORE STORAGE MANAGER ===

class FirestoreStorageManager:
    """Production storage manager using Google Cloud Firestore"""

    def __init__(self):
        """Initialize Firestore client and cache"""
        try:
            self.db = firestore.Client()
            logger.info("Firestore client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Firestore client: {e}")
            logger.warning("Falling back to in-memory storage for development")
            self.db = None

        # Collections
        self.claims_collection = 'claims'
        self.analytics_collection = 'analytics'
        self.regional_facts_collection = 'regional_facts'

        # In-memory cache for performance
        self.cache = SimpleCache(max_size=1000)

        # Initialize analytics document
        self._initialize_analytics()

        logger.info("Firestore storage manager initialized")

    def _initialize_analytics(self):
        """Initialize analytics document if it doesn't exist"""
        if not self.db:
            return

        try:
            analytics_ref = self.db.collection(self.analytics_collection).document('global')
            analytics_doc = analytics_ref.get()

            if not analytics_doc.exists:
                # Create initial analytics document
                initial_analytics = {
                    'total_claims': 0,
                    'verdicts': {
                        'True': 0,
                        'False': 0,
                        'Partially True': 0,
                        'Unverified': 0
                    },
                    'languages': {},
                    'processing_times': [],
                    'last_updated': get_current_timestamp()
                }
                analytics_ref.set(initial_analytics)
                logger.info("Analytics document initialized")
        except Exception as e:
            logger.error(f"Failed to initialize analytics: {e}")

    def save_claim(self, claim_record: ClaimRecord) -> bool:
        """Save claim analysis result to Firestore"""
        try:
            if not self.db:
                logger.warning("Firestore not available, claim not persisted")
                return False

            # Convert to dictionary
            claim_data = asdict(claim_record)

            # Save to Firestore
            claim_ref = self.db.collection(self.claims_collection).document(claim_record.claim_hash)
            claim_ref.set(claim_data)

            # Update analytics atomically
            self._update_analytics(claim_record)

            # Cache the result
            cache_key = f"claim:{claim_record.claim_hash}"
            self.cache.set(cache_key, claim_data, ttl=3600)

            logger.info(f"Claim {claim_record.claim_id} saved to Firestore")
            return True

        except Exception as e:
            logger.error(f"Failed to save claim {claim_record.claim_id}: {e}")
            return False

    def _update_analytics(self, claim_record: ClaimRecord):
        """Update analytics document atomically"""
        try:
            if not self.db:
                return

            analytics_ref = self.db.collection(self.analytics_collection).document('global')

            # Use Firestore transactions for atomic updates
            transaction = self.db.transaction()

            @firestore.transactional
            def update_in_transaction(transaction, analytics_ref):
                snapshot = analytics_ref.get(transaction=transaction)

                if not snapshot.exists:
                    self._initialize_analytics()
                    snapshot = analytics_ref.get(transaction=transaction)

                analytics_data = snapshot.to_dict()

                # Update counters
                analytics_data['total_claims'] = analytics_data.get('total_claims', 0) + 1

                # Update verdict counts
                verdict = claim_record.verdict
                if 'verdicts' not in analytics_data:
                    analytics_data['verdicts'] = {}
                analytics_data['verdicts'][verdict] = analytics_data['verdicts'].get(verdict, 0) + 1

                # Update language counts
                if 'languages' not in analytics_data:
                    analytics_data['languages'] = {}
                language = claim_record.language
                analytics_data['languages'][language] = analytics_data['languages'].get(language, 0) + 1

                # Update processing times (keep last 1000)
                if 'processing_times' not in analytics_data:
                    analytics_data['processing_times'] = []
                analytics_data['processing_times'].append(claim_record.processing_time)
                if len(analytics_data['processing_times']) > 1000:
                    analytics_data['processing_times'] = analytics_data['processing_times'][-1000:]

                analytics_data['last_updated'] = get_current_timestamp()

                transaction.set(analytics_ref, analytics_data)

            update_in_transaction(transaction, analytics_ref)

        except Exception as e:
            logger.error(f"Failed to update analytics: {e}")

    def get_claim(self, claim_hash: str) -> Optional[Dict]:
        """Get claim by hash from Firestore"""
        try:
            # Try cache first
            cache_key = f"claim:{claim_hash}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for claim {claim_hash}")
                return cached_result

            # Query Firestore
            if not self.db:
                return None

            claim_ref = self.db.collection(self.claims_collection).document(claim_hash)
            claim_doc = claim_ref.get()

            if claim_doc.exists:
                claim_data = claim_doc.to_dict()
                # Cache for future requests
                self.cache.set(cache_key, claim_data, ttl=3600)
                logger.debug(f"Claim {claim_hash} retrieved from Firestore")
                return claim_data

            return None

        except Exception as e:
            logger.error(f"Failed to get claim with hash {claim_hash}: {e}")
            return None

    def get_claim_by_id(self, claim_id: str) -> Optional[Dict]:
        """Get claim by claim_id (requires query)"""
        try:
            if not self.db:
                return None

            # Query by claim_id
            claims_ref = self.db.collection(self.claims_collection)
            query = claims_ref.where('claim_id', '==', claim_id).limit(1)
            results = list(query.stream())

            if results:
                claim_data = results[0].to_dict()
                return claim_data

            return None

        except Exception as e:
            logger.error(f"Failed to get claim by ID {claim_id}: {e}")
            return None

    def get_recent_claims(self, limit: int = 50) -> List[Dict]:
        """Get recent claims ordered by created_at"""
        try:
            if not self.db:
                return []

            claims_ref = self.db.collection(self.claims_collection)
            query = claims_ref.order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit)

            claims = []
            for doc in query.stream():
                claims.append(doc.to_dict())

            return claims

        except Exception as e:
            logger.error(f"Failed to get recent claims: {e}")
            return []

    def get_claims_by_verdict(self, verdict: str, limit: int = 100) -> List[Dict]:
        """Get claims filtered by verdict"""
        try:
            if not self.db:
                return []

            claims_ref = self.db.collection(self.claims_collection)
            query = claims_ref.where('verdict', '==', verdict).limit(limit)

            claims = []
            for doc in query.stream():
                claims.append(doc.to_dict())

            return claims

        except Exception as e:
            logger.error(f"Failed to get claims by verdict {verdict}: {e}")
            return []

    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics from Firestore"""
        try:
            if not self.db:
                return {
                    'total_claims_processed': 0,
                    'verdict_distribution': {},
                    'language_distribution': {},
                    'average_processing_time': 0,
                    'cache_size': len(self.cache._cache),
                    'last_updated': get_current_timestamp(),
                    'note': 'Firestore not available'
                }

            analytics_ref = self.db.collection(self.analytics_collection).document('global')
            analytics_doc = analytics_ref.get()

            if not analytics_doc.exists:
                self._initialize_analytics()
                analytics_doc = analytics_ref.get()

            analytics_data = analytics_doc.to_dict()

            # Calculate average processing time
            avg_processing_time = 0
            if analytics_data.get('processing_times'):
                avg_processing_time = sum(analytics_data['processing_times']) / len(analytics_data['processing_times'])

            return {
                'total_claims_processed': analytics_data.get('total_claims', 0),
                'verdict_distribution': analytics_data.get('verdicts', {}),
                'language_distribution': analytics_data.get('languages', {}),
                'average_processing_time': round(avg_processing_time, 2),
                'cache_size': len(self.cache._cache),
                'last_updated': analytics_data.get('last_updated', get_current_timestamp())
            }

        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {}

    def health_check(self) -> HealthStatus:
        """Check system health including Firestore connectivity"""
        try:
            services = {}

            # Check Firestore connectivity
            if self.db:
                try:
                    # Try a simple read operation
                    analytics_ref = self.db.collection(self.analytics_collection).document('global')
                    analytics_ref.get()
                    services['firestore'] = 'healthy'
                except Exception as e:
                    logger.error(f"Firestore health check failed: {e}")
                    services['firestore'] = 'unhealthy'
            else:
                services['firestore'] = 'unavailable'

            # Check cache
            services['cache'] = 'healthy'
            if len(self.cache._cache) > self.cache.max_size * 0.9:
                services['cache'] = 'degraded'

            # Overall status
            if any(status == 'unhealthy' for status in services.values()):
                overall_status = 'unhealthy'
            elif any(status == 'degraded' or status == 'unavailable' for status in services.values()):
                overall_status = 'degraded'
            else:
                overall_status = 'healthy'

            return HealthStatus(
                status=overall_status,
                timestamp=get_current_timestamp(),
                services=services
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthStatus(
                status='unhealthy',
                timestamp=get_current_timestamp(),
                services={'error': str(e)}
            )

    def clear_cache(self) -> bool:
        """Clear cache for testing/debugging"""
        try:
            self.cache.clear()
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def delete_claim(self, claim_hash: str) -> bool:
        """Delete a claim (for admin/testing purposes)"""
        try:
            if not self.db:
                return False

            claim_ref = self.db.collection(self.claims_collection).document(claim_hash)
            claim_ref.delete()

            # Clear from cache
            cache_key = f"claim:{claim_hash}"
            self.cache._cache.pop(cache_key, None)

            logger.info(f"Claim {claim_hash} deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete claim {claim_hash}: {e}")
            return False


# === GLOBAL STORAGE INSTANCE ===

storage_manager = FirestoreStorageManager()


def get_storage() -> FirestoreStorageManager:
    """Get global storage manager instance"""
    return storage_manager


# === UTILITY FUNCTIONS (Backward Compatible) ===

def save_analysis_result(claim_id: str, claim_text: str, claim_hash: str,
                        language: str, verdict: str, confidence_score: int,
                        analysis_result: Dict, processing_time: float) -> bool:
    """Convenience function to save analysis result"""
    try:
        claim_record = ClaimRecord(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_hash=claim_hash,
            language=language,
            verdict=verdict,
            confidence_score=confidence_score,
            analysis_result=analysis_result,
            created_at=get_current_timestamp(),
            processing_time=processing_time
        )

        return storage_manager.save_claim(claim_record)

    except Exception as e:
        logger.error(f"Failed to save analysis result: {e}")
        return False


def get_cached_analysis(claim_hash: str) -> Optional[Dict]:
    """Get cached analysis result"""
    return storage_manager.get_claim(claim_hash)


def get_system_analytics() -> Dict[str, Any]:
    """Get system analytics"""
    return storage_manager.get_analytics()


def get_system_health() -> Dict[str, Any]:
    """Get system health status"""
    health = storage_manager.health_check()
    return asdict(health)


# === ADDITIONAL UTILITY FUNCTIONS ===

def get_recent_claims(limit: int = 50) -> List[Dict]:
    """Get recent claims"""
    return storage_manager.get_recent_claims(limit)


def get_claims_by_verdict(verdict: str, limit: int = 100) -> List[Dict]:
    """Get claims filtered by verdict"""
    return storage_manager.get_claims_by_verdict(verdict, limit)


def get_claim_by_id(claim_id: str) -> Optional[Dict]:
    """Get claim by claim_id"""
    return storage_manager.get_claim_by_id(claim_id)


logger.info("Firestore database module initialized successfully")
