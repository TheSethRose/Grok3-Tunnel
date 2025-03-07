# Grok3 Tunnel

An unofficial (self-hosted) API tunnel that provides access to Grok3 through a simple REST interface.

## ⚠️ Important Disclaimer

**Use at your own risk. Not a polished product.**

1. This is an **unofficial** API tunnel, fragile AF, and will likely break with any Grok updates.
2. Worth noting that I didn't code this - Grok did technically. 😅
3. It DOES create the chats in Grok's web history - all interactions are visible in your Grok account.
4. This project is not affiliated with, endorsed by, or sponsored by xAI, Grok, or any of their affiliates.

This tool is intended for personal use only. Users are responsible for ensuring their use of this tool complies with xAI's [Terms of Service](https://x.ai/legal/terms-of-service). In particular, please note that:

1. You must have a valid Grok account to use this tool
2. You are responsible for all activities conducted through this API
3. Commercial use may be subject to additional restrictions
4. This tool should not be used to develop competing models or services

The creator of this project is not responsible for any misuse of this tool or violations of xAI's Terms of Service.

## Support the Project

<div align="center">

[![Support on GitHub](https://img.shields.io/badge/Sponsor-Support%20my%20work-brightgreen?style=for-the-badge&logo=github)](https://github.com/sponsors/TheSethRose)

</div>

Took guts and caffeine to crank this out. Now, I'm asking YOU to back me. $5 a month. That's it.
Pocket change to keep me fueled and slamming out more projects like this.

Think about it — what's $5? A shitty coffee? Or a ticket to more tools that actually DO something?
I'm not here to beg; I'm here to BUILD. Support me, and I'll keep the good stuff coming. No fluff.

Sponsor me. Let's make more noise together. 🚀 (written by Grok3)

## Quick Start

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Get your Grok cookie string (see [Authentication](#authentication) section)
4. Start the server: `python server.py`
5. The API is available at `http://localhost:11435/api`

## Features

- Simple REST API for interacting with Grok3
- Support for streaming and non-streaming responses
- Special prefixes (@search, @reason) to enable DeepSearch and Thinking modes
- Flexible authentication via cookie string
- Support for image generation and file attachments
- Comprehensive web search capabilities
- Step-by-step reasoning mode

## Requirements

- Python 3.9+
- FastAPI
- Requests
- Valid Grok3 cookie values (requires a valid Grok account)

## Installation

1. Clone this repository
2. Install the dependencies: `pip install -r requirements.txt`
3. Get your Grok cookie string (see Authentication section)

## Authentication

To use the API, you need your Grok cookie string which will serve as your API key in most client applications. Here's how to get it:

1. Go to [Grok.com](https://grok.com)
2. Log in to your account (Grok account required)
3. Open your browser's developer tools (F12)
4. Go to the Network tab
5. Make a request in the Grok UI
6. Find a request to `grok.com`
7. Right-click on the request, select "Copy as cURL"
8. From that cURL command, extract the cookie string

Minimum cookie requirements: `sso` and `sso-rw`

> **Important**: In most API clients (like ChatGPT, Claude, etc.), you should paste your cookie string directly into the API Key field. The cookie string IS your API key.

You can also provide the cookie string in your requests in one of two ways:

1. **Authorization Header (Recommended)**
   ```bash
   Authorization: Bearer your_cookie_string_here
   ```

2. **Request Body**
   ```json
   {
     "cookie_string": "your_cookie_string_here"
   }
   ```

### Client Setup Example

When setting up in clients like ChatGPT or Claude:
1. Provider name: `Grok3`
2. API Base URL: `http://localhost:11435/api`
3. API Key: `your_cookie_string_here` (ex. sso-rw=eyJhbGc...)

## Usage

### Special Prefixes for Advanced Features

This API supports special message prefixes that enable Grok's advanced features without needing to modify the API request structure. This is especially useful when using third-party tools that don't allow customizing the request payload.

Simply add ONE of these prefixes to the beginning of your message:

1. **@search** - Enables DeepSearch mode (comprehensive web search)
   ```
   @search What are the latest developments in quantum computing?
   ```
   This automatically configures:
   - Sets `deepsearchPreset` to "comprehensive"
   - Enables web search tools
   - Sets `disableSearch` to false

2. **@reason** - Enables Thinking/Reasoning mode (step-by-step reasoning)
   ```
   @reason How would you solve this equation: 3x + 5 = 17?
   ```
   This automatically configures:
   - Sets `isReasoning` to true
   - Adds custom instructions for step-by-step reasoning

> **Important**: Use only one prefix at a time (@search OR @reason), not both together. This matches how the official Grok interface works, which only allows using either DeepSearch or Thinking mode in a single query, not both simultaneously.

The prefix will be automatically detected, the corresponding feature will be enabled, and the prefix will be stripped from your message before being sent to Grok.

### API Endpoints

#### Chat Completions Endpoint

The main endpoint for chat interactions:

```python
import requests
import json

# Using the @search prefix to enable DeepSearch
response = requests.post(
    "http://localhost:11435/api/chat/completions",
    headers={
        "Authorization": "Bearer your_cookie_string_here"
    },
    json={
        "model": "grok-3",
        "messages": [{"role": "user", "content": "@search What is the latest research on fusion energy?"}]
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

For streaming:

```python
import requests
import json

# Using the @reason prefix to enable Thinking mode
response = requests.post(
    "http://localhost:11435/api/chat/completions",
    headers={
        "Authorization": "Bearer your_cookie_string_here"
    },
    json={
        "model": "grok-3",
        "messages": [{"role": "user", "content": "@reason Explain how quantum computers work"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith("data: "):
            data = json.loads(line_str[6:])
            if "choices" in data and data["choices"]:
                if "delta" in data["choices"][0]:
                    content = data["choices"][0]["delta"].get("content", "")
                    if content:
                        print(content, end="")
```

#### Generate Endpoint

For simpler use cases, you can use the `/api/generate` endpoint:

```python
import requests
import json

# Using the @search prefix
response = requests.post(
    "http://localhost:11435/api/generate",
    headers={
        "Authorization": "Bearer your_cookie_string_here"
    },
    json={
        "model": "grok-3",
        "prompt": "@search What are the latest advancements in renewable energy?",
        "stream": False
    }
)

print(response.json()["response"])
```

Using curl:

```bash
# Streaming request with @search prefix
curl http://localhost:11435/api/generate \
  -H "Authorization: Bearer your_cookie_string_here" \
  -d '{
    "model": "grok-3",
    "prompt": "@search Why is the sky blue?"
  }'

# Non-streaming request with @reason prefix
curl http://localhost:11435/api/generate \
  -H "Authorization: Bearer your_cookie_string_here" \
  -d '{
    "model": "grok-3",
    "prompt": "@reason Why is the sky blue?",
    "stream": false
  }'
```

### Available Endpoints

- `/api/chat/completions` - Chat completions API
- `/api/chat` - Legacy endpoint for backward compatibility
- `/api/models` - List available models (only returns Grok-3)
- `/api/generate` - Generate API for simple text completion

### Advanced Configuration

The API supports various configuration options through its request payload. Here are the key parameters and their effects:

#### Core Parameters
- `message`: Your input query
- `modelName`: Always "grok-3"
- `disableSearch`: Controls web search capability
- `isReasoning`: Enables step-by-step reasoning
- `deepsearchPreset`: Configures search depth ("comprehensive" for DeepSearch)

#### Image-Related Parameters
- `enableImageGeneration`: Allows image generation
- `imageAttachments`: For including images
- `returnImageBytes`: Get images as byte data
- `enableImageStreaming`: Stream generated images
- `imageGenerationCount`: Number of images to generate

#### Additional Features
- `customInstructions`: Custom behavior guidance
- `sendFinalMetadata`: Include sources and metadata
- `toolOverrides`: Fine-tune available tools
- `forceConcise`: Request shorter responses

## Contributing

### Ways to Contribute

- **Report Bugs**: Found something that doesn't work? Open an issue with detailed steps to reproduce.
- **Suggest Features**: Have ideas for improvements? I'd love to hear them.
- **Submit Pull Requests**: Code contributions are always welcome. Feel free to fork the repo and submit PRs.
- **Documentation**: Help improve or clarify the documentation.
- **Share Use Cases**: Let me know how you're using this tool in interesting ways.

### Development Guidelines

- Keep it simple and maintainable
- Test your changes thoroughly
- Respect the existing code style
- Document any new functionality

If you're unsure about anything, just open an issue to start the conversation. All skill levels are welcome - whether you're fixing a typo or implementing a major feature.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

The MIT License is a permissive license that allows for reuse with few restrictions. It permits users to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that the original copyright notice and permission notice appear in all copies.

### What this means for you:
- You can use this code for personal or commercial projects
- You can modify the code as needed
- You must include the original license and copyright notice
- The software is provided "as is" without warranty

### Copyright Notice
Copyright (c) 2024 Seth Rose - applies only to the original code implementation of this API tunnel.

This copyright applies solely to the original code and implementation created for this project, not to any xAI/Grok intellectual property, APIs, or services. This project is an unofficial interface and does not claim any rights to xAI's proprietary technology, services, or branding.

Users must comply with xAI's Terms of Service when using this tool to access Grok services.
