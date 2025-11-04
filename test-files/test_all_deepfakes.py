"""
Test all deepfake types (audio, image, video) against deployed backend
"""

import requests
import json
import time
from pathlib import Path

# Deployed backend URL
BACKEND_URL = "https://satyacheck-ai-290073510140.us-central1.run.app/"

def test_deepfake_file(file_path, file_type, expected_result):
    """Test a single file against the deployed backend"""

    print(f"\n{'='*70}")
    print(f"Testing: {file_path}")
    print(f"Type: {file_type.upper()}")
    print(f"Expected: {'DEEPFAKE' if expected_result else 'AUTHENTIC'}")
    print('='*70)

    # Prepare the file
    files = {}
    data = {
        'deep_analysis': 'true',
        'language': 'en'
    }

    # Set the correct field name based on file type
    if file_type == 'image':
        files['image_data'] = open(file_path, 'rb')
    elif file_type == 'audio':
        files['audio_data'] = open(file_path, 'rb')
    elif file_type == 'video':
        files['video_data'] = open(file_path, 'rb')

    try:
        start_time = time.time()

        # Make the request
        print(f"Sending request to {BACKEND_URL}...")
        response = requests.post(
            BACKEND_URL,
            files=files,
            data=data,
            timeout=180  # 3 minute timeout for video processing
        )

        elapsed = time.time() - start_time

        # Close the file
        if file_type == 'image':
            files['image_data'].close()
        elif file_type == 'audio':
            files['audio_data'].close()
        elif file_type == 'video':
            files['video_data'].close()

        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Time: {elapsed:.2f}s")

        if response.status_code == 200:
            result = response.json()

            # Extract deepfake result from nested data structure
            data = result.get('data', {})
            is_deepfake = data.get('is_deepfake', False)
            confidence = data.get('confidence', 0)
            verdict = data.get('verdict', data.get('result', 'UNKNOWN'))
            risk_score = data.get('risk_score', 0)

            # Determine if test passed
            test_passed = (is_deepfake == expected_result)

            print(f"\n[RESULT]")
            print(f"   Detected: {'DEEPFAKE' if is_deepfake else 'AUTHENTIC'}")
            print(f"   Confidence: {confidence*100:.1f}%")
            print(f"   Verdict: {verdict}")
            print(f"   Risk Score: {risk_score:.1f}/100")
            print(f"   Test: {'PASS' if test_passed else 'FAIL'}")

            # Show analysis details if available
            if 'deepfake_analysis' in data:
                analysis = data['deepfake_analysis']
                print(f"\n[ANALYSIS]")
                print(f"   Method: {analysis.get('method', 'N/A')}")
                print(f"   Model: {analysis.get('model', 'N/A')}")
                if 'indicators' in analysis and analysis['indicators']:
                    print(f"   Indicators: {', '.join(analysis['indicators'][:3])}")

            # Show processing time
            if 'processing_time' in data:
                print(f"   Processing Time: {data['processing_time']}s")

            return {
                'success': True,
                'file': file_path,
                'type': file_type,
                'expected': expected_result,
                'detected': is_deepfake,
                'test_passed': test_passed,
                'confidence': confidence,
                'risk_score': risk_score,
                'time': elapsed,
                'response': result
            }
        else:
            print(f"\n[ERROR] HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}")

            return {
                'success': False,
                'file': file_path,
                'type': file_type,
                'error': f"HTTP {response.status_code}",
                'test_passed': False
            }

    except requests.exceptions.Timeout:
        print(f"\n[ERROR] Request timeout after 180s")
        return {
            'success': False,
            'file': file_path,
            'type': file_type,
            'error': 'Timeout',
            'test_passed': False
        }

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        return {
            'success': False,
            'file': file_path,
            'type': file_type,
            'error': str(e),
            'test_passed': False
        }


def main():
    print("\n" + "="*70)
    print("TESTING DEPLOYED DEEPFAKE DETECTION SYSTEM")
    print(f"Backend URL: {BACKEND_URL}")
    print("="*70)

    # Define test cases
    test_cases = [
        ('deepfake-image-fake.jpg', 'image', True),
        ('deepfake-image-real.jpg', 'image', False),
        ('deepfake-audio-fake.mp3', 'audio', True),
        ('deepfake-audio-real.mp3', 'audio', False),
        ('deepfake-video-fake.mp4', 'video', True),
        ('deepfake-video-real.mp4', 'video', False),
    ]

    # Check if files exist
    print("\nChecking test files...")
    for file_path, _, _ in test_cases:
        if Path(file_path).exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")

    # Run tests
    results = []
    for file_path, file_type, expected in test_cases:
        if not Path(file_path).exists():
            print(f"\n[SKIP] {file_path} - File not found")
            continue

        result = test_deepfake_file(file_path, file_type, expected)
        results.append(result)

        # Brief pause between tests
        time.sleep(1)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total = len(results)
    passed = sum(1 for r in results if r['test_passed'])
    failed = total - passed
    successful_requests = sum(1 for r in results if r['success'])

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Successful Requests: {successful_requests}")
    print(f"Accuracy: {(passed/total)*100:.1f}%")

    # Breakdown by type
    print("\nBREAKDOWN BY TYPE:")
    for media_type in ['image', 'audio', 'video']:
        type_results = [r for r in results if r.get('type') == media_type]
        if type_results:
            type_passed = sum(1 for r in type_results if r.get('test_passed'))
            type_total = len(type_results)
            avg_time = sum(r.get('time', 0) for r in type_results if r.get('success')) / max(1, sum(1 for r in type_results if r.get('success')))
            print(f"  {media_type.upper()}: {type_passed}/{type_total} passed, Avg time: {avg_time:.2f}s")

    # Failed tests details
    failed_tests = [r for r in results if not r['test_passed']]
    if failed_tests:
        print("\nFAILED TESTS:")
        for r in failed_tests:
            print(f"  - {r['file']}: {r.get('error', 'Detection mismatch')}")

    print("="*70)

    # Save results
    output_file = 'deployed_test_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'backend_url': BACKEND_URL,
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'accuracy': (passed/total)*100 if total > 0 else 0
            },
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Overall verdict
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
    elif passed >= total * 0.8:
        print("\n[WARNING] Most tests passed, but some failed")
    else:
        print("\n[FAILURE] Multiple test failures")

    print()

if __name__ == "__main__":
    main()
