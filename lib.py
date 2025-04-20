# GeminiLib.py
# Compatible with Hikka v1.6.3+

import google.generativeai as genai
import google.ai.generativelanguage as glm
import logging
import re
import random
import os
from PIL import Image
from io import BytesIO
from typing import Union, Optional, Tuple, List, Dict, Any

from .. import loader # Hikka's loader for Library base class
from .. import utils # For run_sync if needed, escape_html
from telethon.extensions import markdown, html
from telethon.tl.types import Message

logger = logging.getLogger(__name__)

# Constants (can be adjusted)
SUPPORTED_PHOTO_MIMES = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]
SUPPORTED_VIDEO_MIMES = ["video/mp4", "video/mpeg", "video/mov", "video/avi", "video/x-flv", "video/mpg", "video/webm", "video/wmv", "video/3gpp"]
SUPPORTED_AUDIO_MIMES = ["audio/wav", "audio/mp3", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac"]
MAX_RESPONSE_LENGTH_FOR_BLOCKQUOTE = 500 # Characters for blockquote wrapping
# Telegram's approx message length limit (be conservative)
TELEGRAM_MSG_LIMIT = 4000

class GeminiLib(loader.Library):
    """Library for interacting with Google Gemini API"""
    developer = "@yg_modules_via_Lib" # You can change this

    def __init__(self):
        # Libraries usually don't need config directly managed here
        # Module passes necessary values like keys, model name
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        # Proxy setup might be needed here if called directly,
        # but better handled by the module setting environment variables.

    # --- Text Processing ---

    def sanitise_text(self, text: str) -> str:
        """Removes Hikka-specific emoji tags."""
        return re.sub(r"</?emoji.*?>", "", text)

    def markdown2html(self, text: str) -> str:
        """Converts Markdown (as understood by Gemini) to HTML for Telegram."""
        try:
            # Basic Markdown to HTML conversion suitable for Telegram
            # Replacing **bold** -> <b>bold</b>
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            # Replacing *italic* -> <i>italic</i> (ensure not part of **)
            # This regex is tricky, might need refinement depending on Gemini's output
            text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
             # Replacing ```code block``` -> <pre><code>code block</code></pre>
            text = re.sub(r'```([\s\S]*?)```', r'<pre><code>\1</code></pre>', text)
            # Replacing `inline code` -> <code>inline code</code>
            text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
            # Handle lists (basic conversion, might need improvement)
            text = re.sub(r'^\s*[\*\-]\s+(.*)', r'• \1', text, flags=re.MULTILINE)
            # Escape remaining HTML special chars
            # text = utils.escape_html(text) # This would escape the tags we just added!
            # Let Hikka handle final parsing, just do basic common markdown.
            return text
        except Exception:
            logger.exception("Markdown to HTML conversion failed, returning raw.")
            # Fallback: return raw text, maybe escape it
            return utils.escape_html(text)


    def _format_response_text(self, text: str) -> str:
        """Converts Gemini Markdown & wraps long text in blockquote."""
        html_text = self.markdown2html(text)
        # Use len(text) for character count, not bytes, for blockquote decision
        if len(text) > MAX_RESPONSE_LENGTH_FOR_BLOCKQUOTE:
            # Ensure blockquote content is properly escaped if needed
            # Assuming markdown2html handles basic escaping within tags
            return f"<blockquote expandable>{html_text}</blockquote>"
        else:
            return html_text

    # --- Media Handling ---

    async def get_media_info(self, message: Message) -> Optional[Tuple[bytes, str, str]]:
        """Extracts supported media bytes, mime_type, and a simple type string."""
        if not message or not message.media:
            return None

        mime_type = None
        media_type_str = None
        downloader = None
        media_to_download = None

        if hasattr(message.media, 'document'):
            doc = message.media.document
            mime_type = getattr(doc, 'mime_type', None)
            if mime_type:
                if mime_type in SUPPORTED_PHOTO_MIMES:
                    media_type_str = "Photo"
                    media_to_download = message
                elif mime_type in SUPPORTED_VIDEO_MIMES:
                    media_type_str = "Video"
                    media_to_download = message
                elif mime_type in SUPPORTED_AUDIO_MIMES:
                    media_type_str = "Audio"
                    media_to_download = message

        elif hasattr(message.media, 'photo'):
            mime_type = "image/jpeg" # Assume JPEG
            media_type_str = "Photo"
            media_to_download = message.media.photo # Download the photo object directly
        elif hasattr(message.media, 'video'):
            mime_type = getattr(message.media.video, 'mime_type', 'video/mp4')
            if mime_type in SUPPORTED_VIDEO_MIMES:
                media_type_str = "Video"
                media_to_download = message
        elif hasattr(message.media, 'voice'):
             mime_type = getattr(message.media.voice, 'mime_type', 'audio/ogg')
             if mime_type in SUPPORTED_AUDIO_MIMES:
                 media_type_str = "Voice" # Keep as Voice for clarity
                 media_to_download = message
        # Add elif for message.audio if needed


        if media_to_download and mime_type and media_type_str:
            try:
                # Download using run_sync if needed, or directly if awaitable
                # The `download_media` method on Message or Photo object is awaitable
                media_bytes = await self._client.download_media(media_to_download, bytes)
                logger.info(f"Lib downloaded {media_type_str} ({mime_type}), size: {len(media_bytes)} bytes")
                return media_bytes, mime_type, media_type_str
            except Exception as e:
                 logger.exception("Library failed to download media")
                 raise RuntimeError(f"Failed to download media: {e}") from e
        else:
            if mime_type: # Found media but unsupported
                raise ValueError(f"Unsupported MIME type: {mime_type}")
            return None # No downloadable/supported media found

    # --- Gemini API Interaction ---

    def _configure_gemini(self, api_keys: List[str], proxy: Optional[str]):
        """Selects an API key and configures GenAI."""
        if not api_keys:
            raise ValueError("No API keys provided in the configuration.")
        api_key = random.choice(api_keys)
        # Proxy setup (should ideally be done once by the module)
        # If proxy provided here, set environment variables if not already set
        if proxy and not os.environ.get("https_proxy"):
            logger.info(f"Lib configuring proxy: {proxy}")
            os.environ["http_proxy"] = proxy
            os.environ["https_proxy"] = proxy
            # Genai might use specific variables or http_proxy/https_proxy
        elif not proxy and os.environ.get("https_proxy"):
             # If no proxy passed but env var exists (from module), clear it? Risky.
             # Best practice: module sets proxy env vars before importing/using lib.
             pass

        genai.configure(api_key=api_key)
        logger.debug(f"Lib configured Gemini with a selected API key.")


    async def call_gemini(
        self,
        api_keys: List[str],
        model_name: str,
        system_instruction: Optional[str],
        proxy: Optional[str],
        prompt: Optional[str],
        media_bytes: Optional[bytes] = None,
        media_mime_type: Optional[str] = None,
    ) -> str:
        """Calls Gemini for a single turn (text or multimodal)."""
        self._configure_gemini(api_keys, proxy)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction or None, # Pass None if empty string
            safety_settings=self.safety_settings,
        )

        content = []
        if media_bytes and media_mime_type:
            content.append(glm.Blob(mime_type=media_mime_type, data=media_bytes))
            logger.debug(f"Lib added media Blob (MIME: {media_mime_type})")

        if prompt:
            content.append(prompt)

        if not content:
            raise ValueError("Cannot call Gemini with empty content.")

        logger.debug(f"Lib calling Gemini model '{model_name}'")
        try:
            response = await model.generate_content_async(
                content,
                safety_settings=self.safety_settings,
                # stream=True # Consider streaming for long responses later
            )
            logger.debug(f"Lib Gemini raw response: {response}")

            if not response.parts:
                logger.warning(f"Lib Gemini response blocked/empty. Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                reason = getattr(response, 'candidates', [None])[0].finish_reason if getattr(response, 'candidates', []) else 0
                reason_map = { 1: "Stop", 2: "Max Tokens", 3: "Safety", 4: "Recitation", 5: "Other"}
                reason_text = reason_map.get(reason, f"Code {reason}")
                # Don't format error message here, return specific error or code
                raise RuntimeError(f"Blocked/Empty response. Reason: {reason_text}")

            # Format the text part of the response
            return response.text.strip() # Return raw text, module formats

        except Exception as e:
            logger.exception("Error calling Gemini API in Lib")
            api_error = getattr(e, 'message', str(e))
            if "API key not valid" in api_error:
                api_error += " Check keys in config."
            # Re-raise a clean error for the module to catch
            raise RuntimeError(f"Gemini API Error: {api_error}") from e

    async def call_gemini_chat(
        self,
        api_keys: List[str],
        model_name: str,
        system_instruction: Optional[str],
        proxy: Optional[str],
        history: List[Dict[str, Any]], # Expects [{"role": "user/model", "parts": [str]}]
        new_prompt: str,
    ) -> str:
        """Calls Gemini with chat history."""
        self._configure_gemini(api_keys, proxy)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction or None,
            safety_settings=self.safety_settings,
        )

        # Start a chat session with existing history
        chat = model.start_chat(history=history)
        logger.debug(f"Lib starting chat with {len(history)} history turns.")

        try:
            # Send the new message
            response = await chat.send_message_async(
                new_prompt,
                safety_settings=self.safety_settings,
            )
            logger.debug(f"Lib Gemini chat response: {response}")

            if not response.parts:
                 logger.warning(f"Lib Gemini chat response blocked/empty. Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                 reason = getattr(response, 'candidates', [None])[0].finish_reason if getattr(response, 'candidates', []) else 0
                 reason_map = { 1: "Stop", 2: "Max Tokens", 3: "Safety", 4: "Recitation", 5: "Other"}
                 reason_text = reason_map.get(reason, f"Code {reason}")
                 raise RuntimeError(f"Blocked/Empty response. Reason: {reason_text}")

            return response.text.strip() # Return raw text

        except Exception as e:
            logger.exception("Error during Gemini chat session in Lib")
            api_error = getattr(e, 'message', str(e))
            if "API key not valid" in api_error:
                api_error += " Check keys in config."
            raise RuntimeError(f"Gemini API Error: {api_error}") from e

    async def list_models(self, api_keys: List[str], proxy: Optional[str]) -> List[Dict[str, Any]]:
        """Lists available Gemini models supporting content generation."""
        self._configure_gemini(api_keys, proxy)
        try:
            # list_models is sync, use run_sync from the library's perspective
            # Note: Requires utils to be available or pass run_sync function
            # For simplicity, assume utils is accessible via `from .. import utils` if lib is standard Hikka lib structure
            models_iterator = await utils.run_sync(genai.list_models)
            available_models = []
            for m in models_iterator:
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append({
                        "name": m.name,
                        "description": m.description
                    })
            return sorted(available_models, key=lambda x: x['name'])
        except Exception as e:
            logger.exception("Error listing models in Lib")
            api_error = getattr(e, 'message', str(e))
            if "API key not valid" in api_error:
                api_error += " Check keys in config."
            raise RuntimeError(f"Model Listing Error: {api_error}") from e