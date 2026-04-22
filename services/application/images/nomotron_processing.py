from typing import List

import cv2
import numpy as np
from ninja import UploadedFile

from services.application.images.pre_processing_base import ImagePreProcessingBase
from services.domain.image_processor import ImageProcessor
from services.domain.prompt import get_semantic_prompt
from services.domain.vaiv_caller import VaivCaller
from services.schema.receipt_output import ItemListModel


class NemotronPreProcessing(ImagePreProcessingBase):
    def __init__(self):
        self.vaiv_caller_nemotron = VaivCaller(provider="vllm")
        self.vaiv_caller_ollama = VaivCaller(provider="ollama")

    def extract_text_with_ocr(
        self, uploaded_files: List[UploadedFile]
    ) -> tuple[ItemListModel, list[str]]:
        imgs = self._load_multi_images(uploaded_files)

        b64s = []
        result_list = []
        constructed_text = ""
        for i, img in enumerate(imgs):
            image = (
                ImageProcessor(img=img)
                .to_grayscale_with_gamma(gamma=1.5)
                .enhance_contrast(clip_limit=2.0)
                .denoise()
                # .sharpen(alpha=1.7)
                .adaptive_binarize(block_size=31)
                .get_image()
            )

            b64 = self._to_base64(image, cv2.COLOR_GRAY2BGR)
            b64s.append(b64)

            img_path = f"./images/{uploaded_files[i].name}"
            self._save_image(image, img_path)

            try:
                ocr_result = self.vaiv_caller_nemotron.call_nemotron(img_path)
                constructed_text += self.reconstruct_text(ocr_result)

            finally:
                self._delete_file(img_path)

        return self._extract_semantic(constructed_text), b64s

    def _save_file(self, uploaded_file: UploadedFile) -> str:
        filename = uploaded_file.name
        file_path = f"./images/{filename}"

        with open(file_path, "wb+") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        return file_path

    @staticmethod
    def reconstruct_text(result: dict, conf_threshold: float = 0.45) -> str:
        raw = result.get("raw", [])
        if not raw:
            return result.get("text", "")

        heights = []
        items = []

        for r in raw:
            text = r.get("text", "").strip()
            conf = r.get("confidence", 0)
            left = r.get("left", 0)
            upper = r.get("upper", 0)
            right = r.get("right", 0)
            lower = r.get("lower", 0)

            if not text or conf < conf_threshold:
                continue

            cy = (upper + lower) / 2.0
            h = max(0.0, lower - upper)
            heights.append(h)
            items.append(
                {
                    "text": text,
                    "conf": conf,
                    "left": left,
                    "right": right,
                    "upper": upper,
                    "lower": lower,
                    "cy": cy,
                    "h": h,
                }
            )

        if not items:
            return result.get("text", "")

        items.sort(key=lambda x: (x["cy"], x["left"]))

        median_h = float(np.median(heights)) if heights else 0.02
        line_tol = max(median_h * 0.6, 0.012)

        lines = []
        current = [items[0]]
        current_y = items[0]["cy"]

        for item in items[1:]:
            if abs(item["cy"] - current_y) <= line_tol:
                current.append(item)
                current_y = (current_y * (len(current) - 1) + item["cy"]) / len(current)
            else:
                lines.append(current)
                current = [item]
                current_y = item["cy"]
        lines.append(current)

        merged_lines = []
        for line in lines:
            line = sorted(line, key=lambda x: x["left"])
            parts = [line[0]["text"]]

            for prev, cur in zip(line, line[1:]):
                gap = cur["left"] - prev["right"]
                if gap > median_h * 1.2:
                    parts.append("  " + cur["text"])
                else:
                    parts.append(" " + cur["text"])

            merged_lines.append("".join(parts).strip())

        return "\n".join(merged_lines)

    def _extract_semantic(self, ocr_result: str) -> ItemListModel:
        return self.vaiv_caller_ollama.call_llm(
            get_semantic_prompt(), ocr_result, None, "qwen3.6:35b-a3b"
        )
