
import os
import asyncio

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

GROQ_API_URL = os.getenv(
    "GROQ_API_URL",
    "https://api.groq.com/openai/v1/chat/completions"
)

REST_API_URL = os.getenv(
    "REST_API_URL",
    "https://jsonplaceholder.typicode.com/posts"
)

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# --------------------------------------------------
# 2. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="FastAPI REST API and Groq AI Integration Lab",
    description="Consume a paginated REST API and analyze its data using Groq AI.",
    version="1.0.0"
)


# --------------------------------------------------
# 3. REQUEST MODEL
# --------------------------------------------------

class AnalyzeRequest(BaseModel):
    page_size: int = Field(default=5, ge=1, le=100)
    max_pages: int = Field(default=3, ge=1, le=20)


# --------------------------------------------------
# 4. COMMON HTTP REQUEST FUNCTION WITH RETRIES
# --------------------------------------------------

async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs
):
    """
    Sends an HTTP request.

    Retries:
    - HTTP 429
    - HTTP 500
    - HTTP 502
    - HTTP 503
    - HTTP 504
    - Network errors
    - Timeout errors

    MAX_RETRIES=3 means:
    1 initial attempt + 3 retries = 4 attempts.
    """

    for attempt in range(MAX_RETRIES + 1):

        try:
            print(f"Request attempt: {attempt + 1}")

            response = await client.request(
                method=method,
                url=url,
                **kwargs
            )

            # Check whether the response status is retryable.
            if response.status_code in RETRYABLE_STATUS_CODES:

                if attempt < MAX_RETRIES:
                    wait_time = 2 ** attempt

                    # Respect Retry-After when supplied as seconds.
                    retry_after = response.headers.get("Retry-After")

                    if retry_after:
                        try:
                            wait_time = max(
                                wait_time,
                                float(retry_after)
                            )
                        except ValueError:
                            pass

                    print(
                        f"HTTP {response.status_code}. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    await asyncio.sleep(wait_time)
                    continue

                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Upstream API failed after retries. "
                        f"HTTP status: {response.status_code}"
                    )
                )

            # Handle other HTTP errors, such as 400 or 401.
            if response.is_error:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Upstream API returned HTTP "
                        f"{response.status_code}: {response.text[:300]}"
                    )
                )

            return response

        except (httpx.TimeoutException, httpx.NetworkError) as error:

            if attempt < MAX_RETRIES:
                wait_time = 2 ** attempt

                print(
                    f"Network/timeout error: {error}. "
                    f"Retrying in {wait_time} seconds..."
                )

                await asyncio.sleep(wait_time)
                continue

            if isinstance(error, httpx.TimeoutException):
                raise HTTPException(
                    status_code=504,
                    detail="Upstream API request timed out after retries."
                )

            raise HTTPException(
                status_code=502,
                detail="Network error while contacting upstream API."
            )

    raise HTTPException(
        status_code=502,
        detail="Request failed after all retry attempts."
    )


# --------------------------------------------------
# 5. FETCH ONE PAGE FROM REST API
# --------------------------------------------------

async def fetch_single_page(
    client: httpx.AsyncClient,
    page: int,
    page_size: int
):
    """
    Fetches one page of posts from JSONPlaceholder.

    Example:
    page=1, page_size=5
    """

    params = {
        "_page": page,
        "_limit": page_size
    }

    response = await request_with_retry(
        client,
        "GET",
        REST_API_URL,
        params=params
    )

    return response.json()


# --------------------------------------------------
# 6. FETCH MULTIPLE PAGES
# --------------------------------------------------

async def fetch_paginated_data(
    page_size: int,
    max_pages: int
):
    """
    Fetches multiple pages from the REST API.
    Stops when:
    - max_pages is reached, or
    - the API returns an empty page.
    """

    all_posts = []

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT
    ) as client:

        for page in range(1, max_pages + 1):

            print(f"Fetching page {page}...")

            posts = await fetch_single_page(
                client,
                page,
                page_size
            )

            if not posts:
                print("No more records found.")
                break

            all_posts.extend(posts)

    return all_posts


# --------------------------------------------------
# 7. CALL GROQ AI API
# --------------------------------------------------

async def generate_ai_summary(posts):

    """
    Sends REST API data to Groq AI
    and returns an AI-generated summary.
    """

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is missing. Configure it in your .env file."
        )

    if not posts:
        return "No posts were available for analysis."

    # Prepare data for the AI model.
    posts_text = ""

    for post in posts:
        posts_text += (
            f"\nPost ID: {post['id']}"
            f"\nTitle: {post['title']}"
            f"\nBody: {post['body']}\n"
        )

    # Prepare Groq chat completion request.
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a data analyst. "
                    "Analyze the supplied posts and provide "
                    "a clear summary, key observations, "
                    "and important patterns."
                )
            },
            {
                "role": "user",
                "content": (
                    "Analyze the following REST API data:\n"
                    f"{posts_text}"
                )
            }
        ],
        "temperature": 0.3,
        "max_completion_tokens": 1000
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT
    ) as client:

        response = await request_with_retry(
            client,
            "POST",
            GROQ_API_URL,
            headers=headers,
            json=payload
        )

    result = response.json()

    # Extract the AI-generated response.
    try:
        ai_summary = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="Unexpected response format received from Groq API."
        )

    return ai_summary


# --------------------------------------------------
# 8. BASIC ENDPOINT
# --------------------------------------------------

@app.get("/")
async def home():

    return {
        "message": "FastAPI REST API and Groq AI Integration Lab",
        "documentation": "/docs"
    }


# --------------------------------------------------
# 9. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
async def health_check():

    return {
        "status": "Application is running",
        "groq_api_key_configured": bool(GROQ_API_KEY),
        "groq_model": GROQ_MODEL
    }


# --------------------------------------------------
# 10. FETCH A SINGLE PAGE
# --------------------------------------------------

@app.get("/posts")
async def get_posts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=5, ge=1, le=100)
):

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT
    ) as client:

        posts = await fetch_single_page(
            client,
            page,
            page_size
        )

    return {
        "page": page,
        "page_size": page_size,
        "total_records": len(posts),
        "data": posts
    }


# --------------------------------------------------
# 11. FETCH ALL PAGINATED DATA
# --------------------------------------------------

@app.get("/posts/fetch-all")
async def get_all_posts(
    page_size: int = Query(default=5, ge=1, le=100),
    max_pages: int = Query(default=3, ge=1, le=20)
):

    posts = await fetch_paginated_data(
        page_size,
        max_pages
    )

    return {
        "message": "Paginated data fetched successfully",
        "total_records": len(posts),
        "data": posts
    }


# --------------------------------------------------
# 12. FETCH REST DATA AND ANALYZE USING GROQ
# --------------------------------------------------

@app.post("/analyze")
async def analyze_posts(request: AnalyzeRequest):

    # Step 1: Fetch paginated REST API data.
    posts = await fetch_paginated_data(
        request.page_size,
        request.max_pages
    )

    # Step 2: Send the data to Groq AI.
    summary = await generate_ai_summary(posts)

    # Step 3: Return the combined result.
    return {
        "message": "REST API data analyzed successfully using Groq AI",
        "total_records": len(posts),
        "model": GROQ_MODEL,
        "ai_summary": summary,
        "data": posts
    }
