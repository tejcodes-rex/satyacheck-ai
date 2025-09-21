"""
SatyaCheck AI - Simplified Database Layer
Minimal database utilities with in-memory storage for hackathon prototype
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from utils import get_current_timestamp, SimpleCache

logger = logging.getLogger(__name__)

# === DATA MODELS ===

@dataclass
class ClaimRecord:
    """Simplified claim record for in-memory storage"""
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

# === SIMPLE STORAGE MANAGER ===

class SimpleStorageManager:
    """Simple in-memory storage for prototype"""
    
    def __init__(self):
        self.claims = {}  # claim_hash -> ClaimRecord
        self.cache = SimpleCache(max_size=500)
        self.analytics = {
            'total_claims': 0,
            'verdicts': {'True': 0, 'False': 0, 'Partially True': 0, 'Unverified': 0},
            'languages': {},
            'processing_times': []
        }
        logger.info("Simple storage manager initialized")
    
    def save_claim(self, claim_record: ClaimRecord) -> bool:
        """Save claim analysis result"""
        try:
            # Store claim
            self.claims[claim_record.claim_hash] = claim_record
            
            # Update analytics
            self.analytics['total_claims'] += 1
            self.analytics['verdicts'][claim_record.verdict] = \
                self.analytics['verdicts'].get(claim_record.verdict, 0) + 1
            self.analytics['languages'][claim_record.language] = \
                self.analytics['languages'].get(claim_record.language, 0) + 1
            self.analytics['processing_times'].append(claim_record.processing_time)
            
            # Cache the result
            cache_key = f"claim:{claim_record.claim_hash}"
            self.cache.set(cache_key, asdict(claim_record), ttl=3600)
            
            logger.info(f"Claim {claim_record.claim_id} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save claim {claim_record.claim_id}: {e}")
            return False
    
    def get_claim(self, claim_hash: str) -> Optional[Dict]:
        """Get claim by hash"""
        try:
            # Try cache first
            cache_key = f"claim:{claim_hash}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Try in-memory storage
            if claim_hash in self.claims:
                claim_record = self.claims[claim_hash]
                result = asdict(claim_record)
                # Cache for future requests
                self.cache.set(cache_key, result, ttl=3600)
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get claim with hash {claim_hash}: {e}")
            return None
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get basic analytics"""
        try:
            avg_processing_time = 0
            if self.analytics['processing_times']:
                avg_processing_time = sum(self.analytics['processing_times']) / len(self.analytics['processing_times'])
            
            return {
                'total_claims_processed': self.analytics['total_claims'],
                'verdict_distribution': self.analytics['verdicts'],
                'language_distribution': self.analytics['languages'],
                'average_processing_time': round(avg_processing_time, 2),
                'cache_size': len(self.cache._cache),
                'last_updated': get_current_timestamp()
            }
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {}
    
    def health_check(self) -> HealthStatus:
        """Check system health"""
        try:
            status = HealthStatus(
                status='healthy',
                timestamp=get_current_timestamp(),
                services={
                    'storage': 'healthy',
                    'cache': 'healthy',
                    'analytics': 'healthy'
                }
            )
            
            # Check storage capacity
            if len(self.claims) > 10000:  # Arbitrary limit for prototype
                status.services['storage'] = 'degraded'
                status.status = 'degraded'
            
            # Check cache
            if len(self.cache._cache) > self.cache.max_size * 0.9:
                status.services['cache'] = 'degraded'
            
            return status
            
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

# === GLOBAL STORAGE INSTANCE ===

storage_manager = SimpleStorageManager()

def get_storage() -> SimpleStorageManager:
    """Get global storage manager instance"""
    return storage_manager

# === UTILITY FUNCTIONS ===

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

logger.info("Database module initialized successfully")
