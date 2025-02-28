import json
import requests
import os
import re
import time
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, UTC
import uvicorn
from dotenv import load_dotenv
import logging
import asyncio
from typing import AsyncGenerator

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_cookie_string(cookie_string):
    """Parse a cookie string into a dictionary of cookie values."""
    if not cookie_string:
        return {}

    cookies = {}

    # Handle different curl formats
    if "-H 'cookie:" in cookie_string:
        # Handle -H 'cookie: format
        cookie_string = cookie_string.split("cookie:", 1)[1]
    elif "-b '" in cookie_string:
        # Handle -b 'cookie_string' format
        cookie_string = cookie_string.split("-b '", 1)[1].split("'", 1)[0]

    # Remove any remaining quotes
    cookie_string = cookie_string.strip("' ")

    # Split the cookie string by semicolons and extract key-value pairs
    for cookie in cookie_string.split(";"):
        if "=" in cookie:
            key, value = cookie.strip().split("=", 1)
            cookies[key] = value

    return cookies

# Initialize empty default cookies (authentication required via API)
DEFAULT_COOKIES = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str  # Only grok-3 is supported
    messages: List[ChatMessage]
    cookies: Optional[Dict[str, str]] = None  # Optional custom cookies (deprecated)
    cookie_string: Optional[str] = None  # Optional cookie string
    stream: Optional[bool] = False  # Optional boolean to enable streaming (default: false)

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    stream: Optional[bool] = True

class GrokClient:
    def __init__(self, cookies: dict):
        self.base_url = "https://grok.com/rest/app-chat/conversations/new"
        self.cookies = cookies
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-GB,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://grok.com",
            "priority": "u=1, i",
            "referer": "https://grok.com/",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "user-agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            )
        }

    def _process_message_prefixes(self, message: str) -> tuple[str, bool, bool]:
        """Process message prefixes and return the cleaned message and feature flags.

        Returns:
            tuple: (cleaned_message, enable_search, enable_reasoning)
        """
        message = message.strip()
        enable_search = False
        enable_reasoning = False

        # Check for prefixes
        has_search = "@search" in message
        has_reason = "@reason" in message

        # Only allow one prefix at a time, prioritizing @search if both are present
        if has_search:
            enable_search = True
            message = message.replace("@search", "", 1).strip()
            logger.info("🔍 @search prefix detected - enabling DeepSearch")
        elif has_reason:
            enable_reasoning = True
            message = message.replace("@reason", "", 1).strip()
            logger.info("🤔 @reason prefix detected - enabling Think mode")

        if enable_search or enable_reasoning:
            logger.info(f"Message prefix processed: DeepSearch={enable_search}, Think={enable_reasoning}")

        return message, enable_search, enable_reasoning

    def _prepare_payload(self, message: str):
        # Process message prefixes
        message, enable_search, enable_reasoning = self._process_message_prefixes(message)

        # Set up tool overrides based on search flag
        tool_overrides = {}
        if enable_search:
            tool_overrides = {
                "webSearch": True,
                "xSearch": True,
                "xMediaSearch": True,
                "trendsSearch": True
            }

        # Set deepsearch preset if search is enabled
        deepsearch_preset = "comprehensive" if enable_search else ""

        # Set custom instructions for reasoning mode
        custom_instructions = ""
        if enable_reasoning:
            custom_instructions = "Break down your thinking step-by-step and show your reasoning process."

        return {
            "temporary": False,
            "modelName": "grok-3",
            "message": message,
            "parentResponseId": "",
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": not enable_search,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": tool_overrides,
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "customInstructions": custom_instructions,
            "deepsearchPreset": deepsearch_preset,
            "isReasoning": enable_reasoning
        }

    async def _async_stream_generator(self, response) -> AsyncGenerator[bytes, None]:
        """Convert sync response to async generator."""
        for line in response.iter_lines():
            if not line:
                continue
            yield line

    async def stream_send_message(self, message: str) -> AsyncGenerator[bytes, None]:
        """Send a message and stream response."""
        payload = self._prepare_payload(message)

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                cookies=self.cookies,
                json=payload,
                stream=True
            )
        except requests.RequestException as e:
            error_msg = {"error": f"Request failed: {e}"}
            error_sse = f"data: {json.dumps(error_msg)}\n\n"
            yield error_sse.encode('utf-8')
            return

        if response.status_code != 200:
            error_msg = {"error": f"Status code {response.status_code}"}
            error_sse = f"data: {json.dumps(error_msg)}\n\n"
            yield error_sse.encode('utf-8')
            return

        final_sent = False
        role_sent = False

        async for line in self._async_stream_generator(response):
            try:
                decoded_line = line.decode('utf-8')
                json_data = json.loads(decoded_line)
                result = json_data.get("result", {})
                response_data = result.get("response", {})

                # Send role delta if not sent yet and we have a valid token
                if not role_sent and response_data.get("token"):
                    current_time = datetime.now(UTC)
                    initial_chunk = {
                        "id": f"chatcmpl-{current_time.strftime('%Y%m%d%H%M%S')}",
                        "object": "chat.completion.chunk",
                        "created": int(current_time.timestamp()),
                        "model": "grok-3",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None
                            }
                        ]
                    }
                    sse_data = f"data: {json.dumps(initial_chunk)}\n\n"
                    yield sse_data.encode('utf-8')
                    role_sent = True

                # If a token is provided, yield it as a partial message
                token = response_data.get("token", "")
                if token:
                    current_time = datetime.now(UTC)
                    response_chunk = {
                        "id": f"chatcmpl-{current_time.strftime('%Y%m%d%H%M%S')}",
                        "object": "chat.completion.chunk",
                        "created": int(current_time.timestamp()),
                        "model": "grok-3",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None
                            }
                        ]
                    }
                    sse_data = f"data: {json.dumps(response_chunk)}\n\n"
                    yield sse_data.encode('utf-8')

                # If the final response is available, yield it and break
                if "modelResponse" in response_data and not final_sent:
                    current_time = datetime.now(UTC)
                    final_chunk = {
                        "id": f"chatcmpl-{current_time.strftime('%Y%m%d%H%M%S')}",
                        "object": "chat.completion.chunk",
                        "created": int(current_time.timestamp()),
                        "model": "grok-3",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    sse_data = f"data: {json.dumps(final_chunk)}\n\n"
                    yield sse_data.encode('utf-8')
                    final_sent = True
                    break
            except json.JSONDecodeError:
                continue
            except Exception:
                continue

    def send_message(self, message: str):
        """Send a message and get a non-streaming response."""
        logger.info("Starting send_message (non-streaming)")
        payload = self._prepare_payload(message)
        # Reduce verbose logging of the entire payload
        logger.info(f"Sending request with message: {message[:50]}...")

        try:
            logger.info(f"Sending request to {self.base_url}")
            response = requests.post(
                self.base_url,
                headers=self.headers,
                cookies=self.cookies,
                json=payload,
                stream=True  # Still use streaming to collect the full response
            )
            logger.info(f"Grok API response status: {response.status_code}")
        except requests.RequestException as e:
            error_msg = {"error": f"Request failed: {e}"}
            logger.error(f"Grok API error: {error_msg}")
            return error_msg

        if response.status_code != 200:
            error_msg = {"error": f"Status code {response.status_code}"}
            logger.error(f"Grok API error: {error_msg}")
            return error_msg

        # Collect the full response
        full_content = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                json_data = json.loads(line.decode('utf-8'))
                result = json_data.get("result", {})
                response_data = result.get("response", {})
                token = response_data.get("token", "")
                if token:
                    full_content += token
            except json.JSONDecodeError:
                continue

        # Return response format
        current_time = datetime.now(UTC)

        response_json = {
            "id": f"chatcmpl-{current_time.strftime('%Y%m%d%H%M%S')}",
            "object": "chat.completion",
            "created": int(current_time.timestamp()),
            "model": "grok-3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(message) // 4,  # Rough estimate
                "completion_tokens": len(full_content) // 4,  # Rough estimate
                "total_tokens": (len(message) + len(full_content)) // 4  # Rough estimate
            }
        }

        # Log summary of the response
        logger.info(f"Non-streaming response completed. Content length: {len(full_content)} chars")

        return response_json

app = FastAPI(
    title="Grok3 API Server",
    description="A REST API server for Grok3.",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to log requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log detailed request information
    client_host = request.client.host if request.client else "Unknown"

    # Extract useful headers, especially those related to the client
    important_headers = {
        "user-agent": request.headers.get("user-agent", "Unknown"),
        "content-type": request.headers.get("content-type", "Unknown"),
        "accept": request.headers.get("accept", "Unknown"),
        "origin": request.headers.get("origin", "Unknown"),
        "referer": request.headers.get("referer", "Unknown"),
    }

    logger.info(f"Request: {request.method} {request.url} from {client_host}")
    logger.info(f"Client headers: {important_headers}")

    # Process the request
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Log response information
    logger.info(f"Response: status={response.status_code}, time={process_time:.2f}s")

    return response

@app.post("/api/chat/completions", summary="Chat completions API")
async def chat_completion(request: ChatRequest, http_request: Request):
    """
    Chat completions API for Grok3.

    **Request Body**:
    - **model**: Only grok-3 is supported.
    - **messages**: Array of chat messages. The last user message is used.
    - **cookie_string**: Optional cookie string for authentication.
    - **stream**: Optional boolean to enable streaming (default: false)
    """
    # Extract user message - get the LAST user message from the conversation
    user_messages = [msg for msg in request.messages if msg.role.lower() == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided.")
    user_message = user_messages[-1].content  # Get the last user message

    # Pretty print the user message
    print("\n" + "─" * 80)
    print("🧑 User:")
    print("└─", user_message)

    # Check for cookie string in different places (in order of priority)
    cookie_string = None

    # 1. Check Authorization header first
    auth_header = http_request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        cookie_string = auth_header[7:]  # Remove "Bearer " prefix

    # 2. Check request cookie_string if no auth header
    if not cookie_string and request.cookie_string:
        cookie_string = request.cookie_string

    # 3. Check request cookies if no cookie_string (backward compatibility)
    if cookie_string:
        cookies = parse_cookie_string(cookie_string)
    elif request.cookies:
        cookies = request.cookies
    else:
        raise HTTPException(
            status_code=401,
            detail="No authentication provided. Please provide a cookie string via Authorization: Bearer header or cookie_string in request body."
        )

    # Check if we have the required cookies
    required_cookies = ["sso", "sso-rw"]
    if not all(k in cookies for k in required_cookies):
        raise HTTPException(
            status_code=401,
            detail=f"Missing required Grok3 cookies. Please provide at least: {', '.join(required_cookies)}"
        )

    client = GrokClient(cookies)

    # Check if streaming is requested
    stream = getattr(request, 'stream', False)

    # Set up SSE headers for streaming responses
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no"  # Disable buffering in Nginx
    }

    # Create a response collector for non-streaming responses
    collected_response = {"content": ""}

    async def collect_response(generator):
        async for chunk in generator:
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode('utf-8')
                if chunk_str.startswith("data: "):
                    try:
                        data = json.loads(chunk_str[6:])
                        if "choices" in data and data["choices"]:
                            if "delta" in data["choices"][0]:
                                content = data["choices"][0]["delta"].get("content", "")
                                if content:
                                    collected_response["content"] += content
                    except json.JSONDecodeError:
                        pass
            yield chunk

        # Print the complete response
        if collected_response["content"]:
            print("\n🤖 Grok:")
            print("└─", collected_response["content"])
            print("─" * 80 + "\n")

    if stream:
        generator = client.stream_send_message(user_message)
        return StreamingResponse(collect_response(generator), headers=headers)
    else:
        response = client.send_message(user_message)
        if isinstance(response, dict) and "choices" in response and response["choices"]:
            first_choice = response["choices"][0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message", {})
                content = message.get("content", "")
                print("\n🤖 Grok:")
                print("└─", content)
                print("─" * 80 + "\n")
        return response

# Keep the old endpoint for backward compatibility
@app.post("/api/chat", summary="Chat with Grok3 (legacy endpoint)")
async def chat_endpoint(request: ChatRequest, http_request: Request):
    """Legacy endpoint. Use /api/chat/completions instead."""
    return await chat_completion(request, http_request)

@app.get("/api/models", summary="List available models")
async def list_models():
    """
    Get information about the models available through the API.
    Currently only returns Grok-3 as it's the only supported model.
    """
    return {
        "data": [
            {
                "id": "grok-3",
                "object": "model",
                "created": 1709251200,  # March 2024 (approximate)
                "owned_by": "xAI",
                "permission": [],
                "root": "grok-3",
                "parent": None,
                "context_window": 8192,  # This is an estimate
                "model_details": {
                    "name": "grok-3",
                    "model": "grok-3",
                    "modified_at": "2024-03-01T00:00:00Z",  # Approximate
                    "details": {
                        "family": "grok",
                        "families": ["grok"],
                        "parameter_size": "Unknown",
                        "capabilities": {
                            "chat_completion": True,
                            "streaming": True
                        }
                    }
                }
            }
        ],
        "object": "list"
    }

@app.get("/api/models/{model_id}", summary="Get model information")
async def get_model(model_id: str):
    """
    Get information about a specific model.
    Currently only supports 'grok-3'.
    """
    if model_id.lower() != "grok-3":
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "id": "grok-3",
        "object": "model",
        "created": 1709251200,  # March 2024 (approximate)
        "owned_by": "xAI",
        "permission": [],
        "root": "grok-3",
        "parent": None,
        "context_window": 8192,  # This is an estimate
        "model_details": {
            "name": "grok-3",
            "model": "grok-3",
            "modified_at": "2024-03-01T00:00:00Z",  # Approximate
            "details": {
                "family": "grok",
                "families": ["grok"],
                "parameter_size": "Unknown",
                "capabilities": {
                    "chat_completion": True,
                    "streaming": True
                }
            }
        }
    }

@app.post("/api/generate", summary="Generate API")
async def generate(request: GenerateRequest, http_request: Request):
    """
    Simple generate API for Grok3.

    **Request Body**:
    - **model**: Only grok-3 is supported
    - **prompt**: The prompt to generate a response for
    - **system**: System message (optional)
    - **stream**: If false the response will be returned as a single object, default: true
    """
    # Log important request details
    client_info = {
        "ip": http_request.client.host if http_request.client else "Unknown",
        "user-agent": http_request.headers.get("user-agent", "Unknown"),
        "content-type": http_request.headers.get("content-type", "Unknown"),
    }

    logger.info(f"Generate request from client: {client_info}")

    # Extract and log key parameters instead of the whole request
    prompt_preview = request.prompt[:50] + "..." if len(request.prompt) > 50 else request.prompt
    system_preview = None
    if request.system:
        system_preview = request.system[:50] + "..." if len(request.system) > 50 else request.system

    request_summary = {
        "model": request.model,
        "prompt_preview": prompt_preview,
        "system_preview": system_preview,
        "stream": request.stream,
    }

    logger.info(f"Generate request parameters: {request_summary}")

    # Check if we have required cookies
    cookies = DEFAULT_COOKIES
    required_cookies = ["sso", "sso-rw"]
    if not cookies or not all(k in cookies for k in required_cookies):
        raise HTTPException(
            status_code=401,
            detail=f"Missing required Grok3 cookies. Please provide at least: {', '.join(required_cookies)}"
        )

    client = GrokClient(cookies)

    # Check if streaming is requested (default is True)
    stream = request.stream if request.stream is not None else True

    # Create a chat request from the generate request
    chat_request = ChatRequest(
        model="grok-3",
        messages=[ChatMessage(role="user", content=request.prompt)],
        stream=stream
    )

    # Use the chat completion endpoint
    return await chat_completion(chat_request, http_request)

if __name__ == "__main__":
    port = 11435
    logger.info(f"Starting Grok3 API Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
