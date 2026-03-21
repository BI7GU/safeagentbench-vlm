import base64
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image


DEFAULT_BASE_URL = "https://api.360.cn/v1/chat/completions"
DEFAULT_MODEL = "alibaba/qwen3-vl-plus"

logger = logging.getLogger(__name__)


class RemoteVLMError(Exception):
    def __init__(self, message, error_type="unknown"):
        super().__init__(message)
        self.error_type = error_type


def get_required_env(name):
    value = os.getenv(name)
    if value:
        return value
    raise RemoteVLMError(
        f"Missing required environment variable: {name}. "
        f"Please export {name} before calling the remote VLM."
    )


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_frame_to_temp_image(frame, suffix=".png"):
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = temp_file.name
    temp_file.close()
    Image.fromarray(frame).save(temp_path)
    logger.debug("Saved frame to temp image: %s", temp_path)
    return temp_path


class RemoteVLMClient:
    def __init__(
        self,
        api_key=None,
        base_url=None,
        model=None,
        timeout=60,
        user="andy",
        max_retries=2,
        retry_delay=1.5,
    ):
        self.api_key = api_key or get_required_env("VLM_API_SECRET_KEY")
        self.base_url = base_url or os.getenv("VLM_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.getenv("VLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.user = user
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _build_payload(self, image_path, user_prompt, temperature=0.2, max_tokens=512):
        image_base64 = encode_image_to_base64(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        }]
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.5,
            "top_k": 0,
            "repetition_penalty": 1.05,
            "num_beams": 1,
            "user": self.user,
            "content_filter": False,
        }

    def generate_from_image_path(self, image_path, user_prompt, temperature=0.2, max_tokens=512):
        image_path = str(Path(image_path))
        payload = self._build_payload(
            image_path=image_path,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }

        logger.info("Calling remote VLM model=%s image=%s", self.model, image_path)
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                logger.info("Remote VLM call succeeded")
                return content
            except requests.ReadTimeout as exc:
                last_error = RemoteVLMError("Remote VLM read timeout", error_type="read_timeout")
            except requests.ConnectTimeout as exc:
                last_error = RemoteVLMError("Remote VLM connection timeout", error_type="connect_timeout")
            except requests.ConnectionError as exc:
                last_error = RemoteVLMError("Remote VLM connection error", error_type="connection_error")
            except requests.RequestException as exc:
                body = exc.response.text if exc.response is not None else str(exc)
                last_error = RemoteVLMError(f"Remote VLM request failed: {body}", error_type="request_error")
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                logger.exception("Remote VLM response parsing failed")
                raise RemoteVLMError("Remote VLM response parsing failed", error_type="response_parse_error") from exc

            logger.warning(
                "Remote VLM call failed on attempt %s/%s: %s",
                attempt + 1,
                self.max_retries + 1,
                last_error,
            )
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        raise last_error

    def generate_from_frame(self, frame, user_prompt, temperature=0.2, max_tokens=512):
        temp_path = save_frame_to_temp_image(frame)
        try:
            return self.generate_from_image_path(
                image_path=temp_path,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        finally:
            try:
                os.remove(temp_path)
                logger.debug("Removed temp image: %s", temp_path)
            except OSError:
                logger.warning("Failed to remove temp image: %s", temp_path)


def gpt4_shadow_with_image(image_path, user_prompt, timeout=60):
    client = RemoteVLMClient(timeout=timeout)
    return client.generate_from_image_path(image_path=image_path, user_prompt=user_prompt)


__all__ = [
    "RemoteVLMClient",
    "RemoteVLMError",
    "encode_image_to_base64",
    "gpt4_shadow_with_image",
    "save_frame_to_temp_image",
]
