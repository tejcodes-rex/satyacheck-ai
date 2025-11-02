"""
SatyaCheck AI - Gemini AI Fact-Checking Service
Integrates Google Gemini for intelligent claim verification
"""

import logging
import json
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)


class GeminiFactChecker:
    """
    Gemini AI-powered fact-checking service.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini service"""
        self.api_key = api_key or GEMINI_API_KEY

        if not self.api_key:
            logger.error("Gemini API key not configured")
            self.enabled = False
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.enabled = True
            logger.info("✅ Gemini AI service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.enabled = False

    def verify_claim_with_sources(
        self,
        claim: str,
        sources: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify claim using Gemini AI with provided sources.

        Args:
            claim: The claim to verify
            sources: List of sources with snippets
            context: Additional context (politician info, regional data)

        Returns:
            Verification result with verdict, explanation, confidence
        """
        if not self.enabled:
            return self._fallback_response()

        try:
            # Build comprehensive prompt
            prompt = self._build_verification_prompt(claim, sources, context)

            # Call Gemini API
            response = self.model.generate_content(prompt)

            # Parse response
            result = self._parse_gemini_response(response.text)

            logger.info(f"Gemini verification complete: {result['verdict']}")
            return result

        except Exception as e:
            logger.error(f"Gemini verification error: {e}", exc_info=True)
            return self._fallback_response()

    def _build_verification_prompt(
        self,
        claim: str,
        sources: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build comprehensive prompt for Gemini"""

        # Format sources for prompt
        sources_text = "\n\n".join([
            f"Source {i+1}: {source.get('name', 'Unknown')}\n"
            f"URL: {source.get('url', 'N/A')}\n"
            f"Trust Score: {source.get('trust_score', 0.5):.2f}\n"
            f"Content: {source.get('snippet', 'No content available')[:500]}"
            for i, source in enumerate(sources[:10])  # Limit to top 10 sources
        ])

        # Add context if available
        context_text = ""
        if context:
            if context.get('politician'):
                pol = context['politician']
                context_text += f"\n\nPolitician Context:\n"
                context_text += f"Name: {pol.get('name')}\n"
                context_text += f"Position: {pol.get('current_position')}\n"
                context_text += f"Party: {pol.get('party')}\n"

            if context.get('state'):
                context_text += f"\nRegional Context: {context['state']}\n"

        prompt = f"""You are a professional fact-checker for SatyaCheck AI, specializing in Indian politics, government claims, and news verification.

CLAIM TO VERIFY:
"{claim}"
{context_text}

SOURCES PROVIDED:
{sources_text if sources else "No sources available."}

INSTRUCTIONS:
1. Analyze the claim carefully
2. Evaluate the provided sources based on their trust scores and content
3. Cross-reference information across multiple sources
4. Consider the credibility of each source
5. Provide a clear verdict with detailed reasoning

RESPONSE FORMAT (JSON):
{{
    "verdict": "true|false|misleading|partly_true|unverifiable|needs_context|satire|opinion",
    "confidence": 0.0-1.0,
    "explanation": "Detailed explanation (200-300 words)",
    "key_facts": ["fact1", "fact2", "fact3"],
    "reasoning": "Step-by-step reasoning process",
    "source_analysis": "Brief analysis of source credibility",
    "warnings": ["warning1 if any"],
    "recommendations": "Additional context needed, if any"
}}

VERDICT DEFINITIONS:
- true: Claim is accurate based on evidence
- false: Claim is demonstrably false
- misleading: Technically true but missing critical context
- partly_true: Some elements are true, others false
- unverifiable: Insufficient evidence to verify
- needs_context: Requires additional context to judge
- satire: Intended as humor/satire
- opinion: Subjective opinion, not factual claim

CONFIDENCE SCORING:
- 0.9-1.0: Multiple high-trust sources confirm (government, fact-checkers)
- 0.7-0.9: Multiple sources agree, some high-trust
- 0.5-0.7: Sources available but conflicting or moderate trust
- 0.3-0.5: Limited sources, low trust
- 0.0-0.3: No reliable sources or highly uncertain

Provide your analysis in JSON format only, no additional text."""

        return prompt

    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's JSON response"""

        try:
            # Try to extract JSON from response
            # Gemini sometimes wraps JSON in markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                json_str = response_text.strip()

            # Parse JSON
            result = json.loads(json_str)

            # Validate required fields
            required = ['verdict', 'confidence', 'explanation']
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Normalize verdict
            result['verdict'] = result['verdict'].lower()

            # Ensure confidence is float between 0-1
            result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")

            # Attempt to extract key information from non-JSON response
            return self._extract_from_text(response_text)

        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            return self._fallback_response()

    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract verdict and explanation from plain text response"""

        text_lower = text.lower()

        # Try to detect verdict from text
        verdict = "unverifiable"
        if any(word in text_lower for word in ['true', 'accurate', 'correct', 'verified']):
            verdict = "true"
        elif any(word in text_lower for word in ['false', 'incorrect', 'wrong', 'fake']):
            verdict = "false"
        elif 'misleading' in text_lower:
            verdict = "misleading"
        elif 'partly' in text_lower or 'partially' in text_lower:
            verdict = "partly_true"

        return {
            'verdict': verdict,
            'confidence': 0.5,
            'explanation': text[:500],  # First 500 chars
            'key_facts': [],
            'reasoning': text[:300],
            'warnings': ['Response format was not standard JSON']
        }

    def _fallback_response(self) -> Dict[str, Any]:
        """Fallback response when Gemini is unavailable"""
        return {
            'verdict': 'unverifiable',
            'confidence': 0.0,
            'explanation': 'Unable to verify claim at this time. Gemini AI service is unavailable.',
            'key_facts': [],
            'reasoning': 'Service unavailable',
            'warnings': ['Gemini AI service is not available']
        }

    def analyze_image_claim(self, image_url: str, claim_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze image for misinformation.
        Uses Gemini Pro Vision for image analysis.
        """
        if not self.enabled:
            return self._fallback_response()

        try:
            # Use Gemini Pro Vision for image analysis
            vision_model = genai.GenerativeModel('gemini-pro-vision')

            prompt = f"""Analyze this image for potential misinformation or manipulation.

{f'Associated Claim: {claim_text}' if claim_text else ''}

Examine:
1. Signs of digital manipulation (photoshop, deepfake)
2. Contextual inconsistencies
3. Reverse image search indicators (old image, different context)
4. Visual artifacts or anomalies

Provide analysis in JSON format:
{{
    "is_manipulated": true/false,
    "confidence": 0.0-1.0,
    "manipulation_signs": ["sign1", "sign2"],
    "context_issues": ["issue1", "issue2"],
    "explanation": "detailed explanation",
    "verdict": "authentic|manipulated|unclear"
}}"""

            # Note: Actual implementation would need to download and pass image
            # For now, return placeholder
            return {
                'verdict': 'unverifiable',
                'confidence': 0.5,
                'explanation': 'Image analysis requires Gemini Pro Vision integration with image download.',
                'key_facts': [],
                'warnings': ['Image analysis not fully implemented yet']
            }

        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return self._fallback_response()

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text (people, places, organizations).
        Useful for politician detection and regional claim routing.
        """
        if not self.enabled:
            return {'people': [], 'places': [], 'organizations': []}

        try:
            prompt = f"""Extract named entities from this text.

TEXT: "{text}"

Provide entities in JSON format:
{{
    "people": ["person1", "person2"],
    "places": ["place1", "place2"],
    "organizations": ["org1", "org2"],
    "dates": ["date1"],
    "politicians_mentioned": ["politician1"]
}}

Focus on Indian politicians, states, cities, and government organizations."""

            response = self.model.generate_content(prompt)

            try:
                # Parse JSON response
                entities = json.loads(response.text.strip())
                return entities
            except:
                # Simple extraction if JSON fails
                return self._simple_entity_extraction(text)

        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {'people': [], 'places': [], 'organizations': []}

    def _simple_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """Simple rule-based entity extraction as fallback"""

        # Common Indian states
        states = ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Gujarat',
                 'Uttar Pradesh', 'West Bengal', 'Rajasthan', 'Kerala', 'Bihar']

        found_places = [state for state in states if state.lower() in text.lower()]

        return {
            'people': [],
            'places': found_places,
            'organizations': [],
            'politicians_mentioned': []
        }


# Global instance
_gemini_service = None


def get_gemini_service() -> GeminiFactChecker:
    """Get global Gemini service instance"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiFactChecker()
    return _gemini_service


logger.info("Gemini fact-checking service module initialized")
