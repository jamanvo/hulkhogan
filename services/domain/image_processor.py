import cv2
import numpy as np
from cv2.typing import MatLike


class ImageProcessor:
    def __init__(self, img: np.ndarray) -> None:
        self._img = img

    def get_image(self) -> np.ndarray:
        if isinstance(self._img, np.ndarray):
            return self._img
        elif isinstance(self._img, MatLike):
            return np.asarray(self._img)

        return self._img

    def to_grayscale_with_gamma(self, gamma: float = 1.2) -> "ImageProcessor":
        """
        BGR → 그레이스케일 변환 후 감마 보정.

        감마 보정 목적:
            카메라 촬영 문서는 과노출(밝은 배경) 또는 저노출(그림자) 영역이
            혼재한다. gamma > 1.0은 어두운 영역을 밝혀 텍스트 대비를 높인다.
            gamma = 1.2는 인쇄 문서 평균 촬영 환경 기준 경험값.
            과노출 이미지가 주로 입력된다면 gamma < 1.0 (예: 0.8)으로 낮춰 쓴다.

        Args:
            img:   BGR 이미지
            gamma: 보정 강도 (1.0 = 보정 없음, >1.0 = 어두운 영역 밝게)

        Returns:
            단일채널 그레이스케일 이미지 (이후 파이프라인은 전부 단일채널)
        """
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

        # LUT(Look-Up Table) 방식: 픽셀별 루프 없이 O(1) 연산
        inv_gamma = 1.0 / gamma
        lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        self._img = cv2.LUT(self._img, lut)

        return self

    def deskew(self) -> "ImageProcessor":
        """
        Hough 라인 기반 기울기 보정. ±1° 미만은 무시(과보정 방지).
        원근 보정 이후이므로 단일채널 입력.
        """
        blurred = cv2.GaussianBlur(self._img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )
        if lines is None:
            return self

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 45:
                    angles.append(angle)

        if not angles:
            return self

        median_angle = np.median(angles)
        if abs(median_angle) < 1.0:
            return self

        h, w = self._img.shape[:2]
        m = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)

        self._img = cv2.warpAffine(
            self._img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        return self

    def enhance_contrast(
        self, clip_limit: float = 3.0, tile_grid_size: tuple[int, int] = (8, 8)
    ) -> "ImageProcessor":
        """
        단일채널 CLAHE.
        v2는 LAB L채널을 사용했으나, 그레이스케일 전환 이후이므로
        직접 단일채널에 적용. clipLimit 3.0 유지.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self._img = clahe.apply(self._img)

        # self._img = cv2.multiply(self._img, 1.5)

        return self

    def enhance_contrast_keep_color(self, clip_limit=2.0, tile_grid=(8, 8)) -> "ImageProcessor":
        lab = cv2.cvtColor(self._img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l = clahe.apply(l)

        self._img = cv2.merge([l, a, b])

        return self

    def denoise(
        self, method: str = "bilateral", d: int = 9, sigma_color: int = 75, sigma_space: int = 75
    ) -> "ImageProcessor":
        """
        단일채널 bilateralFilter.
        엣지 보존하면서 평탄 영역(배경) 평활화.
        """
        if method == "bilateral":
            self._img = cv2.bilateralFilter(
                self._img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
            )

        return self

    def sharpen(
        self,
        alpha: float = 1.8,
        ksize: tuple[int, int] = (0, 0),
        sigma_x: float = 2.0,
        beta: float = -0.8,
        gamma: float = 0,
        sharpening_mask=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    ) -> "ImageProcessor":
        self._img = cv2.filter2D(self._img, -1, sharpening_mask)

        return self

    def adaptive_binarize(self, block_size: int = 21, c: int = 10) -> "ImageProcessor":
        binary = cv2.adaptiveThreshold(
            self._img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=c,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self._img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return self

    def upscale(self, scale_factor: float = 1.5) -> "ImageProcessor":
        width = int(self._img.shape[1] * scale_factor)
        height = int(self._img.shape[0] * scale_factor)
        dimensions = (width, height)

        self._img = cv2.resize(self._img, dimensions, interpolation=cv2.INTER_LANCZOS4)

        return self

    def unsharp_mask(self, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
        # 가우시안 블러를 적용하여 부드러운 이미지 생성
        blurred = cv2.GaussianBlur(self._img, kernel_size, sigma)

        # 원본과 블러 이미지의 차이를 계산하여 윤곽선 강조
        sharpened = float(amount + 1) * self._img - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)

        if threshold > 0:
            low_contrast_mask = np.absolute(self._img - blurred) < threshold
            np.copyto(sharpened, self._img, where=low_contrast_mask)

        self._img = sharpened

        return self

    def rotate_image(self, angle: float = 1.5) -> "ImageProcessor":
        (h, w) = self._img.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        self._img = cv2.warpAffine(
            self._img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        return self
