"""
Secure credential management using Google Cloud Secret Manager.

This module provides secure access to API keys and sensitive configuration
without hardcoding them in the codebase.

RENAMED from secrets.py to secret_manager.py to avoid conflict with Python's built-in secrets module.
"""

import os
import logging
from typing import Optional, Dict
from google.cloud import secretmanager
from google.api_core import exceptions as google_exceptions

# Initialize logger
logger = logging.getLogger(__name__)

# Cache for secrets to avoid repeated API calls
_secrets_cache: Dict[str, str] = {}

# GCP Project ID
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'satyacheck-ai')


def get_secret(secret_id: str, version: str = 'latest') -> Optional[str]:
    """
    Retrieve a secret from Google Cloud Secret Manager.

    Args:
        secret_id: The ID of the secret (e.g., 'gemini-api-key')
        version: The version of the secret to retrieve (default: 'latest')

    Returns:
        The secret value as a string, or None if not found

    Example:
        >>> api_key = get_secret('gemini-api-key')
        >>> if api_key:
        >>>     print("API key retrieved successfully")
    """
    # Check cache first
    cache_key = f"{secret_id}:{version}"
    if cache_key in _secrets_cache:
        logger.debug(f"Retrieved secret '{secret_id}' from cache")
        return _secrets_cache[cache_key]

    try:
        # Create the Secret Manager client
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret version
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/{version}"

        # Access the secret version
        response = client.access_secret_version(request={"name": name})

        # Decode the secret payload
        secret_value = response.payload.data.decode('UTF-8')

        # Cache the secret
        _secrets_cache[cache_key] = secret_value

        logger.info(f"Successfully retrieved secret '{secret_id}' from Secret Manager")
        return secret_value

    except google_exceptions.NotFound:
        logger.error(f"Secret '{secret_id}' not found in Secret Manager")
        return None
    except google_exceptions.PermissionDenied:
        logger.error(f"Permission denied accessing secret '{secret_id}'")
        return None
    except Exception as e:
        logger.error(f"Error accessing secret '{secret_id}': {str(e)}")
        return None


def get_secret_with_fallback(secret_id: str, env_var_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a secret with fallback to environment variable for local development.

    This function tries the following in order:
    1. Google Cloud Secret Manager
    2. Environment variable
    3. Default value

    Args:
        secret_id: The ID of the secret in Secret Manager
        env_var_name: The environment variable name to check as fallback
        default: Default value if neither source is available

    Returns:
        The secret value, or None if not found anywhere

    Example:
        >>> api_key = get_secret_with_fallback(
        >>>     'gemini-api-key',
        >>>     'GEMINI_API_KEY',
        >>>     'default-key-for-dev'
        >>> )
    """
    # Try Secret Manager first (production)
    secret_value = get_secret(secret_id)
    if secret_value:
        return secret_value

    # Fallback to environment variable (local development)
    env_value = os.environ.get(env_var_name)
    if env_value:
        logger.info(f"Using environment variable '{env_var_name}' as fallback")
        return env_value

    # Use default value if provided
    if default:
        logger.warning(f"Using default value for '{secret_id}'")
        return default

    logger.error(f"Could not retrieve secret '{secret_id}' from any source")
    return None


def verify_secret_manager_access() -> bool:
    """
    Verify that the application can access Google Cloud Secret Manager.

    Returns:
        True if Secret Manager is accessible, False otherwise

    Example:
        >>> if verify_secret_manager_access():
        >>>     print("Secret Manager is accessible")
        >>> else:
        >>>     print("Cannot access Secret Manager, using fallback")
    """
    try:
        client = secretmanager.SecretManagerServiceClient()

        # Try to list secrets (this will fail if no permissions)
        parent = f"projects/{PROJECT_ID}"
        request = secretmanager.ListSecretsRequest(parent=parent)

        # Just try to list - we don't need the results
        list(client.list_secrets(request=request, page_size=1))

        logger.info("Secret Manager access verified successfully")
        return True

    except google_exceptions.NotFound:
        logger.warning(f"Project '{PROJECT_ID}' not found in Secret Manager")
        return False
    except google_exceptions.PermissionDenied:
        logger.warning("Permission denied accessing Secret Manager")
        return False
    except Exception as e:
        logger.warning(f"Cannot access Secret Manager: {str(e)}")
        return False


def clear_secrets_cache():
    """
    Clear the secrets cache. Useful for testing or forcing refresh.

    Example:
        >>> clear_secrets_cache()
        >>> # Next get_secret() call will fetch from Secret Manager
    """
    global _secrets_cache
    _secrets_cache.clear()
    logger.info("Secrets cache cleared")


# Convenience functions for common secrets
def get_gemini_api_key() -> Optional[str]:
    """Get Gemini AI API key."""
    return get_secret_with_fallback('gemini-api-key', 'GEMINI_API_KEY')


def get_google_search_api_key() -> Optional[str]:
    """Get Google Custom Search API key."""
    return get_secret_with_fallback('google-search-api-key', 'GOOGLE_SEARCH_API_KEY')


def get_google_search_engine_id() -> Optional[str]:
    """Get Google Custom Search Engine ID."""
    return get_secret_with_fallback('google-search-engine-id', 'GOOGLE_SEARCH_ENGINE_ID')


# Test function for development
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Secret Manager access...")

    # Verify access
    if verify_secret_manager_access():
        print("✅ Secret Manager is accessible")
    else:
        print("❌ Secret Manager is not accessible, will use environment variables")

    # Test retrieving secrets
    print("\nTesting secret retrieval...")

    api_key = get_gemini_api_key()
    if api_key:
        print(f"✅ Gemini API Key: {api_key[:10]}... (length: {len(api_key)})")
    else:
        print("❌ Could not retrieve Gemini API Key")

    search_key = get_google_search_api_key()
    if search_key:
        print(f"✅ Search API Key: {search_key[:10]}... (length: {len(search_key)})")
    else:
        print("❌ Could not retrieve Search API Key")

    engine_id = get_google_search_engine_id()
    if engine_id:
        print(f"✅ Search Engine ID: {engine_id}")
    else:
        print("❌ Could not retrieve Search Engine ID")
