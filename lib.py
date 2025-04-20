# yg_gemini_lib.py

import google.generativeai as genai
import google.ai.generativelanguage as glm
import os
import random
import logging
from typing import List, Optional, Union, Dict, Any

from .. import loader, utils


logger = logging.getLogger(__name__)

# Safety Settings (can be defined within the library)
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

class YgGeminiLib(loader.Library):
    """Library for interacting with Google Gemini API"""

    developer = "@yg_modules" # Or your username

    def __init__(self):
        # Library config can be defined here if needed, but we'll pass values from the module
        self.api_keys = []
        self.model_name = "gemini-1.5-flash-latest"
        self.proxy = None
        self.safety_settings = DEFAULT_SAFETY_SETTINGS
        # No __init__ config using loader.LibraryConfig as settings come from the module

    def config_complete(self):
        # You can add checks here to ensure essential config was passed via init_lib
        # For example: if not self.api_keys: logger.error("Gemini Lib not initialized with API keys!")
        pass

    # Method to receive configuration from the main module
    def init_lib(
        self,
        api_keys: List[str],
        model_name: str,
        proxy: Optional[str] = None,
        safety_settings: Optional[List[Dict[str, str]]] = None,
    ):
        """Initialize the library with settings from the calling module."""
        self.api_keys = api_keys
        self.model_name = model_name
        self.proxy = proxy
        self.safety_settings = safety_settings or DEFAULT_SAFETY_SETTINGS
        self._configure_proxy()
        logger.info(f"YgGeminiLib initialized. Keys loaded: {len(self.api_keys)}, Model: {self.model_name}")

    def _configure_proxy(self):
        """Configure proxy environment variables if proxy is set."""
        if self.proxy:
            logger.info(f"Library configuring proxy: {self.proxy}")
            os.environ["http_proxy"] = self.proxy
            os.environ["https_proxy"] = self.proxy
            os.environ["HTTP_PROXY"] = self.proxy
            os.environ["HTTPS_PROXY"] = self.proxy

    def _get_random_api_key(self) -> str:
        """Selects a random API key from the configured list."""
        if not self.api_keys:
            raise ValueError("No API keys configured for Gemini library.")
        return random.choice(self.api_keys)

    async def call_gemini(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        media_bytes: Optional[bytes] = None,
        media_mime_type: Optional[str] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Calls the Gemini API with optional context, media, and system instruction.

        :param prompt: The user's prompt.
        :param system_instruction: System instruction for the model.
        :param media_bytes: Bytes of the media file.
        :param media_mime_type: MIME type of the media file.
        :param chat_history: List of previous turns [{"role": "user/model", "parts": ["text"]}].
        :return: The raw text response from the model.
        :raises ValueError: If no API key is configured or content is empty.
        :raises RuntimeError: For API-related errors.
        """
        api_key = self._get_random_api_key()
        genai.configure(api_key=api_key) # Configure for this specific call

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction or None,
            safety_settings=self.safety_settings,
        )

        # Prepare content list (history first, then media, then current prompt)
        request_content = []
        if chat_history:
             # Ensure history format matches API (list of Content objects or dicts)
             # The stored history should already be in the correct [{"role": ..., "parts": [...]}] format
            request_content.extend(chat_history)
            logger.debug(f"Prepending chat history with {len(chat_history)} turns.")

        if media_bytes and media_mime_type:
            request_content.append(glm.Blob(mime_type=media_mime_type, data=media_bytes))
            logger.debug(f"Adding media Blob (MIME: {media_mime_type}) to request.")

        if prompt:
            # Append the current user prompt as the last part
            # Gemini expects the last item for the current turn typically as just the parts
            # However, sending as {"role": "user", "parts": [prompt]} is also valid for multi-turn explicitly
            # Let's stick to the format matching history for consistency if history exists.
            # If no history, just the prompt parts suffice.
            if chat_history:
                 request_content.append({"role": "user", "parts": [prompt]})
            else:
                # For single-turn requests, we might not need the role explicitly,
                # especially if media is present. Let's combine media and prompt parts.
                if media_bytes and media_mime_type:
                    # Media blob is already added, just append prompt text part
                     if isinstance(request_content[-1], glm.Blob):
                         # If last item was blob, add prompt as new part
                         request_content.append(prompt)
                     else: # Should not happen based on above logic, but safeguard
                         request_content.append({"role": "user", "parts": [prompt]})
                else:
                    # No media, no history, just the prompt
                    request_content.append(prompt)


        if not request_content:
             raise ValueError("Cannot call Gemini with empty content.")

        logger.debug(f"Calling Gemini model '{self.model_name}' with {len(request_content)} content parts.")

        try:
            # Use the actual history list for the `history` parameter if the model supports it,
            # otherwise send the full content list. Gemini models generally handle the full list well.
            # The `history` param in generate_content is more for starting a ChatSession easily.
            # Sending the full structured list in `contents` is robust.
            response = await model.generate_content_async(
                contents=request_content, # Send the constructed list
                safety_settings=self.safety_settings
            )
            logger.debug(f"Gemini raw response object: {response}")

            # Error/Block Handling
            if not response.parts:
                 logger.warning(f"Gemini response blocked or empty. Prompt Safety Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                 finish_reason = getattr(response, 'candidates', [None])[0].finish_reason if getattr(response, 'candidates', []) else 'UNKNOWN'
                 reason_map = { 1: "Normal Stop", 2: "Max Tokens", 3: "Safety", 4: "Recitation", 5: "Other"}
                 reason_text = reason_map.get(finish_reason, f"Code {finish_reason}")
                 # Return the error message directly
                 return f"⚠️ Response was blocked or empty (Reason: {reason_text})."

            return response.text.strip()

        except Exception as e:
            logger.exception("Error during Gemini API call in library")
            gemini_error_info = getattr(e, 'message', str(e))
            if "API key not valid" in gemini_error_info:
                gemini_error_info += " One of the provided API keys is invalid."
            # Re-raise as RuntimeError for the module to catch
            raise RuntimeError(f"Gemini API Error: {gemini_error_info}") from e

    async def list_models(self) -> List[Dict[str, str]]:
        """Lists available Gemini models supporting content generation."""
        api_key = self._get_random_api_key()
        genai.configure(api_key=api_key)

        try:
            logger.debug("Listing models from Gemini API...")
            # list_models is sync, needs run_sync if called from async context in module
            # If library method is async, we can run it directly here.
            # Let's make the library method async and run sync inside if needed.
            models_iterator = await utils.run_sync(genai.list_models) # Run sync call in executor
            # models_iterator = genai.list_models() # If genai itself becomes async later

            available_models = []
            for m in models_iterator:
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append({
                        "name": m.name,
                        "description": m.description,
                    })
            logger.debug(f"Found {len(available_models)} models supporting generateContent.")
            return available_models

        except Exception as e:
            logger.exception("Error listing models in library")
            gemini_error_info = getattr(e, 'message', str(e))
            if "API key not valid" in gemini_error_info:
                 gemini_error_info += " One of the provided API keys might be invalid."
            raise RuntimeError(f"Failed to list models: {gemini_error_info}") from e