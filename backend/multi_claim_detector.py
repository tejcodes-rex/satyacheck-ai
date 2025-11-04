"""
Multi-Claim Detection Module
"""

import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExtractedClaim:
    text: str
    index: int
    confidence: float
    claim_type: str


class MultiClaimDetector:
    """Detects multiple claims in a single input."""

    def __init__(self):
        self.numbered_pattern = re.compile(r'(?:^|\n)\s*(\d+[\.\):])\s*(.+?)(?=(?:\n\s*\d+[\.\):])|$)', re.DOTALL)
        self.bullet_pattern = re.compile(r'(?:^|\n)\s*[•\-\*]\s*(.+?)(?=(?:\n\s*[•\-\*])|$)', re.DOTALL)
        self.hindi_numbered = re.compile(r'(?:^|\n)\s*([१२३४५६७८९०]+[\.\):])\s*(.+?)(?=(?:\n\s*[१२३४५६७८९०]+[\.\):])|$)', re.DOTALL)

    def detect_claims(self, text: str) -> Dict[str, Any]:
        """Main detection function - returns multiple claims if found."""
        text = text.strip()

        if not text or len(text) < 10:
            return {
                'is_multi_claim': False,
                'total_claims': 0,
                'claims': []
            }

        claims = []
        numbered_claims = self._detect_numbered_claims(text)
        if numbered_claims:
            claims = numbered_claims
            logger.info(f"Detected {len(claims)} numbered claims")

        elif self._has_bullets(text):
            bulleted_claims = self._detect_bulleted_claims(text)
            if bulleted_claims:
                claims = bulleted_claims
                logger.info(f"Detected {len(claims)} bulleted claims")

        elif self._has_multiple_sentences(text):
            sentence_claims = self._detect_sentence_claims(text)
            if len(sentence_claims) >= 2:
                claims = sentence_claims
                logger.info(f"Detected {len(claims)} sentence-based claims")

        if not claims or len(claims) == 1:
            return {
                'is_multi_claim': False,
                'total_claims': 1,
                'claims': [{
                    'text': text,
                    'index': 1,
                    'confidence': 1.0,
                    'claim_type': 'single'
                }]
            }

        claims = [c for c in claims if len(c['text'].strip()) >= 10]

        return {
            'is_multi_claim': len(claims) > 1,
            'total_claims': len(claims),
            'claims': claims
        }

    def _detect_numbered_claims(self, text: str) -> List[Dict]:
        """Detect numbered claims."""
        claims = []
        matches = self.numbered_pattern.findall(text)
        for i, (number, claim_text) in enumerate(matches, 1):
            claims.append({
                'text': claim_text.strip(),
                'index': i,
                'confidence': 0.95,
                'claim_type': 'numbered',
                'number': number
            })

        if not claims:
            hindi_matches = self.hindi_numbered.findall(text)
            for i, (number, claim_text) in enumerate(hindi_matches, 1):
                claims.append({
                    'text': claim_text.strip(),
                    'index': i,
                    'confidence': 0.90,
                    'claim_type': 'numbered_hindi',
                    'number': number
                })

        return claims

    def _detect_bulleted_claims(self, text: str) -> List[Dict]:
        """Detect bulleted claims."""
        claims = []
        matches = self.bullet_pattern.findall(text)

        for i, claim_text in enumerate(matches, 1):
            claims.append({
                'text': claim_text.strip(),
                'index': i,
                'confidence': 0.90,
                'claim_type': 'bulleted'
            })

        return claims

    def _detect_sentence_claims(self, text: str) -> List[Dict]:
        """Detect multiple sentence-based claims."""
        sentences = self._split_sentences(text)
        sentences = [s for s in sentences if len(s.strip()) >= 20]

        claims = []
        for i, sentence in enumerate(sentences, 1):
            if self._is_likely_claim(sentence):
                claims.append({
                    'text': sentence.strip(),
                    'index': i,
                    'confidence': 0.75,
                    'claim_type': 'sentence'
                })

        return claims

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_enders = r'[।\.\!\?]\s+'
        sentences = re.split(sentence_enders, text)
        if len(sentences) <= 1 and '\n' in text:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]

        return sentences

    def _is_likely_claim(self, sentence: str) -> bool:
        """Check if sentence is likely a factual claim."""
        sentence_lower = sentence.lower()

        if any(word in sentence_lower for word in ['hello', 'hi', 'hey', 'namaste', 'नमस्ते']):
            return False

        if sentence.strip().endswith('?'):
            return False

        word_count = len(sentence.split())
        if word_count < 5:
            return False
        claim_keywords = [
            'announced', 'घोषणा', 'said', 'कहा', 'will', 'होगा',
            'scheme', 'योजना', 'government', 'सरकार', 'free', 'मुफ्त'
        ]

        has_claim_keyword = any(kw in sentence_lower for kw in claim_keywords)

        return has_claim_keyword or word_count >= 8

    def _has_bullets(self, text: str) -> bool:
        """Check if text has bullet points."""
        return bool(re.search(r'[•\-\*]\s+', text))

    def _has_multiple_sentences(self, text: str) -> bool:
        """Check if text has multiple sentences."""
        sentence_count = len(re.findall(r'[।\.\!\?]', text))
        return sentence_count >= 2

    def combine_verdicts(self, individual_verdicts: List[Dict]) -> Dict[str, Any]:
        """Combine verdicts from multiple claims into overall verdict."""
        if not individual_verdicts:
            return {
                'overall_verdict': 'Unverified',
                'overall_confidence': 0,
                'summary': 'No claims to verify'
            }

        verdicts = [v.get('verdict', 'Unverified') for v in individual_verdicts]
        confidences = [v.get('score', 0) for v in individual_verdicts]

        false_count = verdicts.count('False')
        true_count = verdicts.count('True')
        partial_count = verdicts.count('Partially True')
        unverified_count = verdicts.count('Unverified')

        total = len(verdicts)

        if false_count >= total * 0.5:
            overall = 'False'
            confidence = sum(confidences) / total
        elif true_count == total:
            overall = 'True'
            confidence = sum(confidences) / total
        elif true_count >= total * 0.7:
            overall = 'Partially True'
            confidence = sum(confidences) / total
        elif unverified_count == total:
            overall = 'Unverified'
            confidence = 50
        else:
            overall = 'Partially True'
            confidence = sum(confidences) / total

        summary = self._generate_summary(verdicts, total)

        return {
            'overall_verdict': overall,
            'overall_confidence': int(confidence),
            'summary': summary,
            'breakdown': {
                'true': true_count,
                'false': false_count,
                'partially_true': partial_count,
                'unverified': unverified_count,
                'total': total
            }
        }

    def _generate_summary(self, verdicts: List[str], total: int) -> str:
        """Generate human-readable summary."""
        false_count = verdicts.count('False')
        true_count = verdicts.count('True')

        if false_count == total:
            return f"All {total} claims are false"
        elif true_count == total:
            return f"All {total} claims are true"
        elif false_count > 0:
            return f"{false_count} out of {total} claims are false"
        else:
            return f"Mixed results: {true_count} true, {total - true_count} need verification"


_multi_claim_detector = None

def get_multi_claim_detector():
    """Get singleton instance."""
    global _multi_claim_detector
    if _multi_claim_detector is None:
        _multi_claim_detector = MultiClaimDetector()
    return _multi_claim_detector
