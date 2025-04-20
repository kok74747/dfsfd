# https://your.host/gemini_lib.py
# -----------------------------------------------------------------------------
# GeminiLib for Hikka Userbot
# Requires: google-generativeai
# -----------------------------------------------------------------------------

import google.generativeai as genai
import google.ai.generativelanguage as glm
import random
import logging
import os
from typing import Optional, List, Dict, Union, Any, Tuple

# Import loader and utils relative to where Hikka loads libraries
from .. import loader, utils

logger = logging.getLogger(__name__)

# Supported MIME types (can be updated if needed)
SUPPORTED_PHOTO_MIMES = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]
SUPPORTED_VIDEO_MIMES = ["video/mp4", "video/mpeg", "video/mov", "video/avi", "video/x-flv", "video/mpg", "video/webm", "video/wmv", "video/3gpp"]
SUPPORTED_AUDIO_MIMES = ["audio/wav", "audio/mp3", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac"]

class GeminiLib(loader.Library):
    developer = "@yg_modules"  # Or your username

    def __init__(self):
        # No LibraryConfig needed here, config is passed from the module
        self.api_keys = []
        self.model_name = "gemini-1.5-flash-latest"
        self.system_instruction = None
        self.proxy = None
        self.client = None # Hikka client instance, set via method
        # Define safety settings once
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    def _set_client(self, client):
        """Allow the main module to pass the client instance"""
        self.client = client

    def configure(
        self,
        api_keys: List[str],
        model_name: str,
        system_instruction: Optional[str],
        proxy: Optional[str],
    ):
        """Configure the library with settings from the main module"""
        self.api_keys = api_keys
        self.model_name = model_name
        self.system_instruction = system_instruction or None # Ensure None if empty string
        self.proxy = proxy

        if self.proxy:
            logger.info(f"[GeminiLib] Using proxy: {self.proxy}")
            os.environ["http_proxy"] = self.proxy
            os.environ["https_proxy"] = self.proxy
            os.environ["HTTP_PROXY"] = self.proxy
            os.environ["HTTPS_PROXY"] = self.proxy
        else:
            # Clear proxy env vars if not set, preventing issues if it was set before
            for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
                if var in os.environ:
                    del os.environ[var]

    def _get_random_api_key(self) -> Optional[str]:
        """Selects a random API key from the configured list."""
        if not self.api_keys:
            return None
        return random.choice(self.api_keys)

    def _format_chat_history(self, history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Formats the simple history list into Gemini's expected format."""
        gemini_history = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                # Gemini expects 'parts', which is a list containing the text
                gemini_history.append({"role": role, "parts": [content]})
        return gemini_history

    async def call_gemini(
        self,
        prompt: str,
        media_bytes: Optional[bytes] = None,
        media_mime_type: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None, # Expects [{"role": "user"/"model", "content": "..."}, ...]
    ) -> str:
        """
        Calls the Gemini API with prompt, optional media, and optional chat history.

        Args:
            prompt: The user's text prompt.
            media_bytes: Optional bytes of the media file.
            media_mime_type: Optional MIME type of the media file.
            chat_history: Optional list of previous chat messages.

        Returns:
            The raw text response from Gemini.

        Raises:
            ValueError: If API key is missing or content is empty.
            RuntimeError: For API communication errors or blocked responses.
        """
        api_key = self._get_random_api_key()
        if not api_key:
            raise ValueError("API key list is empty or not configured.")

        try:
            # Configure genai each time to potentially use a different key/proxy setting
            genai.configure(api_key=api_key)

            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_instruction, # Use configured instruction
                safety_settings=self.safety_settings,
            )

            content_parts = []
            if media_bytes and media_mime_type:
                content_parts.append(glm.Blob(mime_type=media_mime_type, data=media_bytes))
                logger.debug(f"[GeminiLib] Added raw media Blob to content (MIME: {media_mime_type})")

            # Always add the current prompt *last* for multi-turn consistency
            if prompt:
                content_parts.append(prompt)

            if not content_parts: # Safeguard if only media was intended but failed processing
                raise ValueError("Cannot call Gemini with empty content parts (prompt missing?).")

            # Format history if provided
            formatted_history = self._format_chat_history(chat_history) if chat_history else []

            logger.debug(f"[GeminiLib] Calling model '{self.model_name}' with {len(content_parts)} current content parts and {len(formatted_history)} history parts.")

            # Start chat session if history exists
            chat = model.start_chat(history=formatted_history)

            # Send the new message (content_parts) within the chat session
            response = await chat.send_message_async(
                 content_parts,
                 safety_settings=self.safety_settings
             )

            # --- Response Handling ---
            logger.debug(f"[GeminiLib] Gemini raw response object: {response}")

            # Check for blocks or empty responses *before* accessing .text
            try:
                 # Accessing .text can raise ValueError if blocked
                 response_text = response.text.strip()

                 # Check if the response was non-empty after stripping
                 if not response_text:
                     logger.warning(f"[GeminiLib] Gemini response was empty after stripping.")
                     finish_reason_val = getattr(response.candidates[0], 'finish_reason', 0) if response.candidates else 0
                     reason_map = {1: "Normal Stop", 2: "Max Tokens", 3: "Safety", 4: "Recitation", 5: "Other", 0: "Unknown"}
                     reason_text = reason_map.get(finish_reason_val, f"Code {finish_reason_val}")
                     # Provide a more specific message for empty responses
                     raise RuntimeError(f"Gemini returned an empty response (Reason: {reason_text}).")

                 return response_text # Return raw text

            except ValueError as e: # Catch blocks specifically
                 logger.warning(f"[GeminiLib] Gemini response blocked. Error: {e}. Prompt Safety Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                 finish_reason_val = getattr(response.candidates[0], 'finish_reason', 0) if response.candidates else 0
                 # Safety is reason 3
                 if finish_reason_val == 3 or "SAFETY" in str(e).upper():
                     # You might want to inspect response.prompt_feedback.safety_ratings here
                     raise RuntimeError("Ответ был заблокирован Gemini из-за настроек безопасности.")
                 else: # Other reasons (e.g., recitation)
                     reason_map = {1: "Normal Stop", 2: "Max Tokens", 3: "Safety", 4: "Recitation", 5: "Other", 0: "Unknown"}
                     reason_text = reason_map.get(finish_reason_val, f"Code {finish_reason_val}")
                     raise RuntimeError(f"Ответ был заблокирован Gemini или пуст (Причина: {reason_text}).")

        except Exception as e:
            logger.exception("[GeminiLib] Error calling Gemini API")
            # Try to make API key errors clearer
            gemini_error_info = getattr(e, 'message', str(e))
            if "API key not valid" in gemini_error_info or "API_KEY_INVALID" in str(e).upper():
                gemini_error_info += " Проверьте ключи в конфиге модуля или срок их действия."
            # Re-raise as RuntimeError for the main module to catch
            raise RuntimeError(f"{gemini_error_info}") from e


    async def list_models(self) -> List[Dict[str, str]]:
        """Lists available Gemini models capable of content generation."""
        api_key = self._get_random_api_key() # Need a key to list models
        if not api_key:
            raise ValueError("API key list is empty or not configured.")

        try:
            genai.configure(api_key=api_key)
            # list_models is sync, run it in executor if called from async context
            # Assuming Hikka's utils.run_sync handles this
            models = await utils.run_sync(genai.list_models)

            available_models = []
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append({
                        "name": m.name,
                        "description": m.description,
                    })
            return available_models

        except Exception as e:
            logger.exception("[GeminiLib] Error listing models")
            err_str = str(e)
            if "API key not valid" in err_str or "API_KEY_INVALID" in str(e).upper():
                err_str += " Проверьте ключи в конфиге модуля."
            raise RuntimeError(f"Не удалось получить список моделей: {err_str}") from e

    # --- Media Helper ---
    def get_media_info(self, message: Any) -> Optional[Tuple[Any, str, str]]:
        """
        Extracts media downloader, mime_type, and a simple type string from a Telethon message.
        Returns downloader coroutine, mime_type, type_str.
        Raises ValueError on unsupported MIME type.
        """
        if not message or not message.media:
            return None

        mime_type = None
        media_type_str = None
        downloader = None # This will be the coroutine message.download_media(...)

        # Check specific types first (more reliable mime types often)
        if hasattr(message.media, 'photo') and message.photo:
            # Telethon photos don't have a direct mime_type, assume JPEG
            mime_type = "image/jpeg"
            media_type_str = "Photo" # EN base type
            downloader = message.download_media(bytes)
        elif hasattr(message.media, 'video') and message.video:
            mime_type = getattr(message.media.video, 'mime_type', 'video/mp4') # Guess mp4
            if mime_type in SUPPORTED_VIDEO_MIMES:
                media_type_str = "Video"
                downloader = message.download_media(bytes)
        elif hasattr(message.media, 'voice') and message.voice:
            mime_type = getattr(message.media.voice, 'mime_type', 'audio/ogg') # Often ogg
            if mime_type in SUPPORTED_AUDIO_MIMES:
                media_type_str = "Voice"
                downloader = message.download_media(bytes)
        elif hasattr(message.media, 'audio') and message.audio: # Regular audio file
             mime_type = getattr(message.media.audio, 'mime_type', None)
             if mime_type in SUPPORTED_AUDIO_MIMES:
                 media_type_str = "Audio"
                 downloader = message.download_media(bytes)

        # Check document last as a fallback or for other types
        elif hasattr(message.media, 'document'):
            doc = message.media.document
            mime_type = getattr(doc, 'mime_type', None)
            if mime_type:
                if mime_type in SUPPORTED_PHOTO_MIMES:
                     media_type_str = "Photo"
                     downloader = message.download_media(bytes)
                elif mime_type in SUPPORTED_VIDEO_MIMES:
                     media_type_str = "Video"
                     downloader = message.download_media(bytes)
                elif mime_type in SUPPORTED_AUDIO_MIMES:
                     media_type_str = "Audio"
                     downloader = message.download_media(bytes)
                # else: Document is not a supported media type by mime

        # Check if we found a supported downloadable type
        if downloader and mime_type and media_type_str:
             # Check again if the detected mime_type is actually supported
             all_supported = SUPPORTED_PHOTO_MIMES + SUPPORTED_VIDEO_MIMES + SUPPORTED_AUDIO_MIMES
             if mime_type not in all_supported:
                 logger.warning(f"[GeminiLib] Media detected ({media_type_str}) but MIME type '{mime_type}' is not in supported lists.")
                 raise ValueError(f"Unsupported MIME type: {mime_type}")
             return downloader, mime_type, media_type_str # Return coroutine
        else:
            # If mime_type exists but wasn't supported for download
            if mime_type:
                 logger.info(f"[GeminiLib] Media MIME type '{mime_type}' is not supported.")
                 raise ValueError(f"Unsupported MIME type: {mime_type}")
            return None # No downloadable/supported media found