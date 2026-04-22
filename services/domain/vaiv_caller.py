import asyncio
import os
import time
from datetime import datetime

import httpx
import requests
from django.conf import settings


class VaivCaller:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.vaiv_username = settings.VAIV_USERNAME
        self.vaiv_password = settings.VAIV_PASSWORD
        self.httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300, read=300),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                keepalive_expiry=300,
            ),
        )
        self.keycloak_url = None
        self.vaiv_url = None
        self.client_id = None
        self.client_secret = None

        self._set_llm_params()

    def _set_llm_params(self):
        provider = self.provider.upper()

        self.keycloak_url = getattr(settings, f"KEYCLOAK_TOKEN_URL_{provider}")
        self.vaiv_url = getattr(settings, f"VAIV_URL_{provider}")
        self.client_id = getattr(settings, f"CLIENT_ID_{provider}")
        self.client_secret = getattr(settings, f"CLIENT_SECRET_{provider}")

    def get_access_token(self) -> str | None:
        access_token = None
        try:
            token_response = requests.post(
                self.keycloak_url,
                data={
                    "grant_type": "password",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "username": self.vaiv_username,
                    "password": self.vaiv_password,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )

            if token_response.status_code != 200:
                return None

            token_json = token_response.json()
            access_token = token_json.get("access_token")
            if not access_token:
                return None
        except Exception as e:
            print("VAIV Auth Error: ", e)

        return access_token

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        encoded_images: list[str] | None,
        model: str,
        max_retries: int = 2,
    ) -> str | None:
        print(f"{datetime.now()} - VAIV LLM 호출 시작")
        access_token = self.get_access_token()
        headers = self._build_header(access_token)
        payload = self._build_payload(model, system_prompt, user_prompt, encoded_images)

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(self.vaiv_url, headers=headers, json=payload, timeout=300)
                if response.status_code != 200:
                    raise RuntimeError(response.text)

                return self._parse_response(response.json())
            except Exception as e:
                if attempt < max_retries:
                    print(f"재시도 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(1.5)
                else:
                    raise RuntimeError(f"LLM 호출 실패: {e}") from e

        return None

    async def call_llm_async(
        self,
        system_prompt: str,
        user_prompt: str,
        encoded_image: str | None,
        model: str,
        max_retries: int = 2,
    ) -> str | None:
        print(f"{datetime.now()} - VAIV LLM(async) 호출 시작")
        access_token = self.get_access_token()
        headers = self._build_header(access_token)
        payload = self._build_payload(model, system_prompt, user_prompt, encoded_image)

        for attempt in range(max_retries + 1):
            try:
                response = await self.httpx_client.post(
                    self.vaiv_url, headers=headers, json=payload
                )

                print(f"{datetime.now()} - VAIV LLM(async) 호출 종료")
                if response.status_code != 200:
                    raise RuntimeError(response.text)

                return self._parse_response(response.json())
            except Exception as e:
                if attempt < max_retries:
                    print(f"재시도 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(1.5)
                else:
                    raise RuntimeError(f"LLM 호출 실패: {e}") from e

        return None

    @staticmethod
    def _build_header(access_token: str | None) -> dict:
        if not access_token:
            raise ValueError

        return {
            "Content-Type": "application/json",
            "code_company": "HDS",
            "code_service": "image2text",
            "Authorization": f"Bearer {access_token}",
        }

    def _build_payload(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        encoded_images: list[str] | None,
    ) -> dict:
        messages = [{"role": "system", "content": system_prompt}]

        if encoded_images is not None:
            if self.provider == "vllm":
                image_part = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    }
                    for encoded_image in encoded_images
                ]
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                            *image_part,
                        ],
                    }
                )
            else:
                messages.append(
                    {"role": "user", "content": user_prompt, "images": [e for e in encoded_images]}
                )
        else:
            messages.append({"role": "user", "content": user_prompt})

        return {
            "model": self._convert_model_name(model),
            "stream": False,
            "temperature": 0.0,
            "max_tokens": 4096,
            "messages": messages,
            "think": False,
        }

    def _convert_model_name(self, model: str) -> str:
        if model == "gemma4:31b":
            return "gemma4-31b-vllm"

        return model

    def _parse_response(self, response: dict) -> str:
        if self.provider == "vllm":
            return response["choices"][0]["message"]["content"]
        else:
            return response["message"]["content"]

    async def vote(
        self,
        vote_system_prompt: str,
        system_prompt: str,
        user_prompt: str,
        encoded_images: list[str],
        model: str,
    ) -> str | None:
        tasks = [
            self.call_llm_async(system_prompt, user_prompt, encoded_image, model)
            for encoded_image in encoded_images
        ]

        print(f"{datetime.now()} - OCR 시작")
        results = await asyncio.gather(*tasks)

        merged_ocr_result = "".join(
            [f"{i + 1}번 결과: {result}\n" for i, result in enumerate(results)]
        )

        print(f"{datetime.now()} - OCR 결과 투표 시작")
        return self.call_llm(vote_system_prompt, merged_ocr_result, None, model)

    def call_nemotron(self, file_path: str) -> dict:
        access_token = self.get_access_token()

        lower = file_path.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            mime = "image/jpeg"
        else:
            mime = "image/png"

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, mime)}

            start = time.time()
            response = requests.post(
                settings.VAIV_OCR_URL,
                files=files,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=300,
            )
            elapsed = time.time() - start

        print(f"⏱️ OCR 응답 시간: {elapsed:.2f}s ({os.path.basename(file_path)})")

        if response.status_code != 200:
            raise RuntimeError(f"❌ OCR 실패: {response.text}")

        return response.json()
