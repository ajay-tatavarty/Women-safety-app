from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
import time
from collections import defaultdict
from typing import Dict, Tuple

app = FastAPI()

SERVICES = {
    "serviceA": "http://127.0.0.1:8001",
    "serviceB": "http://127.0.0.1:8002"
}

# This is our "Secret Password"
SECRET_KEY = "Ajay123"

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 10  # Number of requests allowed
RATE_LIMIT_WINDOW = 60  # Time window in seconds

# Rate limiting storage: {client_id: (count, window_start_time)}
rate_limit_counters: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, time.time()))


class RateLimiter:
    """Counter-based rate limiter with fixed time window"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.counters: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, time.time()))
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed for the client.
        Returns: (is_allowed, rate_limit_info)
        """
        current_time = time.time()
        count, window_start = self.counters[client_id]
        
        # Check if window has expired
        if current_time - window_start >= self.window_seconds:
            # Reset counter for new window
            self.counters[client_id] = (1, current_time)
            remaining = self.max_requests - 1
            reset_time = int(current_time + self.window_seconds)
            return True, {
                "limit": self.max_requests,
                "remaining": remaining,
                "reset": reset_time
            }
        
        # Check if limit exceeded
        if count >= self.max_requests:
            reset_time = int(window_start + self.window_seconds)
            return False, {
                "limit": self.max_requests,
                "remaining": 0,
                "reset": reset_time
            }
        
        # Increment counter
        self.counters[client_id] = (count + 1, window_start)
        remaining = self.max_requests - (count + 1)
        reset_time = int(window_start + self.window_seconds)
        
        return True, {
            "limit": self.max_requests,
            "remaining": remaining,
            "reset": reset_time
        }


# Initialize rate limiter
rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)


@app.get("/{service_name}/hello")
async def proxy_request(service_name: str, request: Request):
    # --- NEW SECURITY CHECK ---
    user_key = request.headers.get("X-Api-Key")
    
    if user_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid or Missing API Key!")
    # ---------------------------
    
    # --- RATE LIMITING CHECK ---
    # Use API key as client identifier for rate limiting
    client_id = user_key or request.client.host if request.client else "unknown"
    is_allowed, rate_info = rate_limiter.is_allowed(client_id)
    
    if not is_allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Limit: {rate_info['limit']} requests per {RATE_LIMIT_WINDOW} seconds",
                "rate_limit": rate_info
            },
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": str(rate_info["reset"]),
                "Retry-After": str(rate_info["reset"] - int(time.time()))
            }
        )
    # ---------------------------

    if service_name not in SERVICES:
        return {"error": "Service not found"}, 404

    target_url = f"{SERVICES[service_name]}/hello"

    async with httpx.AsyncClient() as client:
        response = await client.get(target_url)
        
        # Add rate limit headers to successful responses
        headers = {
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset"])
        }
        
        return JSONResponse(
            content=response.json(),
            headers=headers
        )