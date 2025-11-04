"""
Enhanced Deepfake Detection System
Specialized models for image, audio, and video deepfake detection
"""

import os
import logging
from typing import Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedImageDetector:
    """Enhanced image deepfake detection using ViT with CLIP fallback."""

    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.processor = None
        self.fallback_to_clip = False

        logger.info("Initializing Enhanced Image Detector")

        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Attempting to load ViT Deepfake Detector v2")
            self.processor = AutoImageProcessor.from_pretrained(
                'prithivMLmods/Deep-Fake-Detector-v2-Model'
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                'prithivMLmods/Deep-Fake-Detector-v2-Model'
            ).to(self.device)

            logger.info(f"ViT model loaded on {self.device}")

        except Exception as e:
            logger.warning(f"ViT model failed to load: {e}")
            logger.info("Falling back to CLIP detector")
            self.fallback_to_clip = True

            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch

                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
                self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

                logger.info(f"CLIP model loaded on {self.device}")

            except Exception as clip_error:
                logger.error(f"CLIP also failed: {clip_error}")
                raise Exception("All image models failed to load")

    def detect(self, image_path: str) -> Dict[str, Any]:
        """Detect deepfake in image."""
        try:
            from PIL import Image
            import torch

            image = Image.open(image_path).convert('RGB')

            if self.fallback_to_clip:
                return self._detect_with_clip(image)
            else:
                return self._detect_with_vit(image)

        except Exception as e:
            logger.error(f"Image detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'is_deepfake': False,
                'confidence': 0.5,
                'risk_score': 50.0,
                'verdict': 'ANALYSIS FAILED'
            }

    def _detect_with_vit(self, image) -> Dict[str, Any]:
        """Use ViT model."""
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            real_score = probs[0][0].item()
            fake_score = probs[0][1].item()

            logger.info(f"Scores - Real: {real_score:.3f}, Fake: {fake_score:.3f}")

            is_deepfake = real_score > fake_score
            confidence = max(real_score, fake_score)
            risk_score = real_score * 100

        if risk_score < 30:
            verdict = 'AUTHENTIC - High confidence'
            risk_level = 'low'
        elif risk_score < 55:
            verdict = 'UNCERTAIN - Manual review recommended'
            risk_level = 'medium'
        elif risk_score < 80:
            verdict = 'LIKELY DEEPFAKE'
            risk_level = 'high'
        else:
            verdict = 'DEEPFAKE DETECTED - High confidence'
            risk_level = 'critical'

        return {
            'success': True,
            'media_type': 'image',
            'is_deepfake': is_deepfake,
            'is_authentic': not is_deepfake,
            'confidence': confidence,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'verdict': verdict,
            'method': 'vit_deepfake_detector_v2',
            'model_score': fake_score
        }

    def _detect_with_clip(self, image) -> Dict[str, Any]:
        """Use CLIP fallback."""
        import torch

        labels = [
            "a real photograph taken by a camera",
            "an AI-generated fake image or deepfake",
        ]

        inputs = self.processor(text=labels, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

        fake_score = float(probs[1])
        is_deepfake = fake_score > 0.50
        confidence = fake_score if is_deepfake else (1 - fake_score)
        risk_score = fake_score * 100

        return {
            'success': True,
            'media_type': 'image',
            'is_deepfake': is_deepfake,
            'is_authentic': not is_deepfake,
            'confidence': confidence,
            'risk_score': risk_score,
            'risk_level': 'medium',
            'verdict': 'DEEPFAKE' if is_deepfake else 'AUTHENTIC',
            'method': 'clip_fallback',
            'note': 'Using CLIP (ViT unavailable)'
        }


class EnhancedAudioDetector:
    """Enhanced audio deepfake detection using Wav2Vec2."""

    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.processor = None
        self.use_forensics = False

        logger.info("Initializing Enhanced Audio Detector")

        try:
            from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Attempting to load Wav2Vec2 Deepfake Detector")
            model_id = 'MelodyMachine/Deepfake-audio-detection-V2'
            self.processor = AutoFeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(self.device)

            logger.info(f"Wav2Vec2 model loaded on {self.device}")

        except Exception as e:
            logger.warning(f"Wav2Vec2 failed to load: {e}")
            logger.info("Falling back to forensic audio analysis")
            self.use_forensics = True

    def detect(self, audio_path: str) -> Dict[str, Any]:
        """Detect audio deepfake."""
        try:
            if self.use_forensics:
                return self._detect_with_forensics(audio_path)
            else:
                return self._detect_with_wav2vec(audio_path)

        except Exception as e:
            logger.error(f"Audio detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'is_deepfake': False,
                'confidence': 0.5,
                'risk_score': 50.0,
                'verdict': 'ANALYSIS FAILED'
            }

    def _detect_with_wav2vec(self, audio_path: str) -> Dict[str, Any]:
        """Use Wav2Vec2 Deepfake Detection model."""
        import librosa
        import torch
        import numpy as np

        audio, sr = librosa.load(audio_path, sr=16000, duration=10)
        if not isinstance(audio, np.ndarray):
            audio_input = np.array(audio, dtype=np.float32)
        else:
            audio_input = audio.astype(np.float32)

        audio_list = audio_input.tolist()
        inputs = self.processor(
            audio_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            real_score = probs[0][0].item()
            fake_score = probs[0][1].item()
        is_deepfake = fake_score > real_score
        confidence = max(real_score, fake_score)
        risk_score = fake_score * 100

        if risk_score < 30:
            verdict = 'AUTHENTIC AUDIO - High confidence'
            risk_level = 'low'
        elif risk_score < 55:
            verdict = 'LIKELY AUTHENTIC AUDIO'
            risk_level = 'medium'
        elif risk_score < 80:
            verdict = 'LIKELY SYNTHETIC AUDIO'
            risk_level = 'high'
        else:
            verdict = 'DEEPFAKE AUDIO - High confidence'
            risk_level = 'critical'

        return {
            'success': True,
            'media_type': 'audio',
            'is_deepfake': is_deepfake,
            'is_authentic': not is_deepfake,
            'confidence': confidence,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'verdict': verdict,
            'method': 'wav2vec2_deepfake_v2_specialized',
            'model': 'Wav2Vec2-Deepfake-V2 (99.73% accuracy)',
            'duration': len(audio) / sr,
            'scores': {
                'real': real_score,
                'fake': fake_score
            }
        }

    def _detect_with_forensics(self, audio_path: str) -> Dict[str, Any]:
        """Simple audio analysis fallback."""
        try:
            import librosa
            import numpy as np

            y, sr = librosa.load(audio_path, sr=16000)

            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            score = 0
            if spectral_centroid > 3000: score += 0.3
            if zcr > 0.15: score += 0.3

            is_fake = score > 0.4
            risk_score = score * 100

            if risk_score < 30:
                verdict = 'AUTHENTIC AUDIO'
                risk_level = 'low'
            elif risk_score < 55:
                verdict = 'LIKELY AUTHENTIC'
                risk_level = 'medium'
            elif risk_score < 80:
                verdict = 'LIKELY SYNTHETIC'
                risk_level = 'high'
            else:
                verdict = 'DEEPFAKE AUDIO'
                risk_level = 'critical'

            return {
                'success': True,
                'is_deepfake': is_fake,
                'confidence': score if is_fake else (1 - score),
                'media_type': 'audio',
                'is_authentic': not is_fake,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'verdict': verdict,
                'method': 'forensic_analysis',
                'note': 'Using forensic analysis (Wav2Vec2 unavailable)'
            }

        except Exception as e:
            logger.error(f"Forensic analysis also failed: {e}")
            return {
                'success': False,
                'error': 'All audio detection methods failed',
                'is_deepfake': False,
                'confidence': 0.5,
                'risk_score': 50.0,
                'verdict': 'ANALYSIS FAILED'
            }


class EnhancedVideoDetector:
    """Video deepfake detection using frame analysis."""

    def __init__(self):
        logger.info("Initializing Enhanced Video Detector")

    def detect(self, video_path: str) -> Dict[str, Any]:
        """Simple video deepfake detection."""
        try:
            import cv2
            import numpy as np

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            frames = []
            for _ in range(5):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()

            if not frames:
                raise ValueError("No frames extracted")

            scores = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                scores.append(1 if blur < 100 else 0)

            avg_score = np.mean(scores)
            is_fake = avg_score > 0.5
            risk_score = avg_score * 100

            if risk_score < 30:
                verdict = 'AUTHENTIC VIDEO'
                risk_level = 'low'
            elif risk_score < 55:
                verdict = 'LIKELY AUTHENTIC'
                risk_level = 'medium'
            elif risk_score < 80:
                verdict = 'LIKELY DEEPFAKE'
                risk_level = 'high'
            else:
                verdict = 'DEEPFAKE VIDEO'
                risk_level = 'critical'

            return {
                'success': True,
                'media_type': 'video',
                'is_deepfake': is_fake,
                'is_authentic': not is_fake,
                'confidence': avg_score if is_fake else (1 - avg_score),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'verdict': verdict,
                'method': 'video_frame_analysis',
                'duration': duration,
                'frames_analyzed': len(frames)
            }

        except Exception as e:
            logger.error(f"Video detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'media_type': 'video',
                'is_deepfake': False,
                'is_authentic': True,
                'confidence': 0.5,
                'risk_score': 50.0,
                'verdict': 'ANALYSIS FAILED'
            }


class EnhancedDeepfakeSystem:
    """Unified deepfake detection system."""

    def __init__(self):
        logger.info("="*70)
        logger.info("ENHANCED DEEPFAKE DETECTION SYSTEM")
        logger.info("="*70)

        self.image_detector = EnhancedImageDetector()
        self.audio_detector = EnhancedAudioDetector()
        self.video_detector = EnhancedVideoDetector()

        logger.info("All detectors initialized")
        logger.info("="*70)

    def analyze(self, file_path: str, media_type: str = None) -> Dict[str, Any]:
        """Analyze any media file for deepfake detection."""
        if not os.path.exists(file_path):
            return {'success': False, 'error': 'File not found'}

        if media_type is None:
            ext = Path(file_path).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                media_type = 'image'
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                media_type = 'video'
            elif ext in ['.mp3', '.wav', '.ogg', '.m4a']:
                media_type = 'audio'
            else:
                return {'success': False, 'error': f'Unsupported file type: {ext}'}

        logger.info(f"\n{'='*70}")
        logger.info(f"📁 Analyzing {media_type.upper()}: {Path(file_path).name}")
        logger.info(f"{'='*70}")

        if media_type == 'image':
            result = self.image_detector.detect(file_path)
        elif media_type == 'audio':
            result = self.audio_detector.detect(file_path)
        elif media_type == 'video':
            result = self.video_detector.detect(file_path)
        else:
            return {'success': False, 'error': f'Unknown media type: {media_type}'}

        self._print_result(result, file_path)
        return result

    def _print_result(self, result: Dict, file_path: str):
        """Print formatted result."""
        print(f"\n{'='*70}")
        print(f"[RESULT] {Path(file_path).name}")
        print(f"{'='*70}")

        if result.get('success'):
            print(f"Verdict: {result.get('verdict', 'UNKNOWN')}")
            print(f"Deepfake: {'YES' if result.get('is_deepfake') else 'NO'}")
            print(f"Confidence: {result.get('confidence', 0)*100:.1f}%")
            print(f"Risk Score: {result.get('risk_score', 0):.1f}/100")
            print(f"Method: {result.get('method', 'unknown')}")
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")

        print(f"{'='*70}\n")

_image_detector_instance = None
_audio_detector_instance = None
_video_detector_instance = None


def detect_image_deepfake(image_path: str) -> Dict[str, Any]:
    """Wrapper function for image deepfake detection."""
    global _image_detector_instance

    try:
        if _image_detector_instance is None:
            _image_detector_instance = EnhancedImageDetector()

        result = _image_detector_instance.detect(image_path)
        return result
    except Exception as e:
        logger.error(f"Image deepfake detection failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'is_deepfake': False,
            'confidence': 0.5,
            'risk_score': 50.0,
            'verdict': 'ANALYSIS FAILED'
        }


def detect_audio_deepfake(audio_path: str) -> Dict[str, Any]:
    """Wrapper function for audio deepfake detection."""
    global _audio_detector_instance

    try:
        if _audio_detector_instance is None:
            _audio_detector_instance = EnhancedAudioDetector()

        result = _audio_detector_instance.detect(audio_path)
        return result
    except Exception as e:
        logger.error(f"Audio deepfake detection failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'is_deepfake': False,
            'confidence': 0.5,
            'risk_score': 50.0,
            'verdict': 'ANALYSIS FAILED'
        }


def detect_video_deepfake(video_path: str) -> Dict[str, Any]:
    """Wrapper function for video deepfake detection."""
    global _video_detector_instance

    try:
        if _video_detector_instance is None:
            _video_detector_instance = EnhancedVideoDetector()

        result = _video_detector_instance.detect(video_path)
        return result
    except Exception as e:
        logger.error(f"Video deepfake detection failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'is_deepfake': False,
            'confidence': 0.5,
            'risk_score': 50.0,
            'verdict': 'ANALYSIS FAILED'
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enhanced_detectors.py <file_path>")
        sys.exit(1)

    system = EnhancedDeepfakeSystem()
    result = system.analyze(sys.argv[1])

    sys.exit(0 if result.get('success') else 1)
