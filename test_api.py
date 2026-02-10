import asyncio
from kalshi_client import KalshiClient
from config import load_config

async def test():
    config = load_config()
    client = KalshiClient(config)
    
    # Test balance
    balance = await client.get_balance()
    print(f"Balance: {balance}¢ (${balance/100:.2f})")
    
    # Test portfolio
    portfolio = await client.get_portfolio_value()
    print(f"Portfolio: {portfolio}¢ (${portfolio/100:.2f})")
    
    await client.close()

asyncio.run(test())