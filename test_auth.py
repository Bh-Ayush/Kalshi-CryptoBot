import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from kalshi_client import KalshiClient

async def test():
    # Create client with credentials from .env
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    
    api_key = os.getenv('KALSHI_API_KEY_ID')
    key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')
    base_url = os.getenv('KALSHI_BASE_URL')
    
    print(f"API Key: {api_key}")
    print(f"Key Path: {key_path}")
    print(f"Base URL: {base_url}")
    print()
    
    # Read and validate private key
    try:
        with open(key_path, 'rb') as f:
            key_data = f.read()
        
        print("Private key file content (first 50 chars):")
        print(key_data[:50].decode('utf-8', errors='ignore'))
        print()
        
        # Try to load the key
        private_key = serialization.load_pem_private_key(
            key_data,
            password=None,
            backend=default_backend()
        )
        print("✓ Private key loaded successfully!")
        print()
        
    except Exception as e:
        print(f"✗ Error loading private key: {e}")
        return
    
    # Try to connect
    print("Testing API connection...")
    try:
        client = KalshiClient(
            api_key=api_key,
            private_key_path=key_path,
            base_url=base_url
        )
        
        balance = await client.get_balance()
        print(f"✓ Balance: {balance}¢ (${balance/100:.2f})")
        
        await client.close()
        
    except Exception as e:
        print(f"✗ API Error: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test())