import cv2
import numpy as np
from typing import Tuple, Optional


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders coordinates in clockwise order:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Applies a perspective transform to obtain a top-down flattened view."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # Compute height of new image
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width <= 0 or max_height <= 0:
        return image

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
    return warped


class DocumentDetector:
    def __init__(self, min_area_ratio: float = 0.15):
        """
        :param min_area_ratio: Minimum ratio of document area relative to full image.
        """
        self.min_area_ratio = min_area_ratio

    def detect_and_align(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detects document boundary and crops/aligns the ROI.
        If detection fails, returns the original image and False.
        """
        if image is None or image.size == 0:
            return image, False

        h, w = image.shape[:2]
        total_area = h * w
        min_area = total_area * self.min_area_ratio

        # Scale down for faster and more reliable contour detection on large images
        scale = 1.0
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            resized = image.copy()

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Strategy 1: Edge detection + morphological close
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in sorted_contours:
                area = cv2.contourArea(contour)
                orig_area = area / (scale * scale)
                if orig_area < min_area:
                    break

                # Exclude contours that span the entire canvas frame (outer border / noise)
                rh, rw = resized.shape[:2]
                bx, by, bw, bh = cv2.boundingRect(contour)
                if (bw >= rw - 4 and bh >= rh - 4) or (orig_area / total_area) > 0.95:
                    continue

                peri = cv2.arcLength(contour, True)
                for eps_factor in (0.02, 0.03):
                    approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        pts = approx.reshape(4, 2).astype("float32") / scale
                        rect = order_points(pts)
                        (tl, tr, br, bl) = rect
                        w_max = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
                        h_max = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
                        if w_max > 0 and h_max > 0:
                            aspect_ratio = max(w_max, h_max) / min(w_max, h_max)
                            if aspect_ratio > 2.5:
                                continue
                            # Only reject near-square contours if they are small inner features (< 25% area)
                            if aspect_ratio < 1.15 and (orig_area / total_area) < 0.25:
                                continue
                            warped = four_point_transform(image, pts)
                            return warped, True

        # Strategy 2: Background color difference mask (handles cards on uniform/colored surfaces)
        rh, rw = resized.shape[:2]
        border_pixels = np.concatenate([
            resized[0:4, :].reshape(-1, 3),
            resized[-4:, :].reshape(-1, 3),
            resized[:, 0:4].reshape(-1, 3),
            resized[:, -4:].reshape(-1, 3)
        ], axis=0) if len(resized.shape) == 3 else np.concatenate([
            resized[0:4, :].reshape(-1, 1),
            resized[-4:, :].reshape(-1, 1),
            resized[:, 0:4].reshape(-1, 1),
            resized[:, -4:].reshape(-1, 1)
        ], axis=0)
        bg_color = np.median(border_pixels, axis=0)
        border_std = float(np.std(border_pixels))

        # Only use background-difference if borders form a uniform/solid background surface (rejects random noise)
        if border_std < 25.0:
            diff = np.linalg.norm(resized.astype("float32") - bg_color, axis=2) if len(resized.shape) == 3 else np.abs(resized.astype("float32") - bg_color[0])

            fg_mask = (diff > 18).astype(np.uint8) * 255
            kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            fg_closed = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_bg)
            cnts_bg, _ = cv2.findContours(fg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if cnts_bg:
                for contour in sorted(cnts_bg, key=cv2.contourArea, reverse=True):
                    area = cv2.contourArea(contour)
                    orig_area = area / (scale * scale)
                    if orig_area < min_area:
                        break

                    bx, by, bw, bh = cv2.boundingRect(contour)
                    # Skip contours that merely span the outer frame edges
                    if bw >= rw - 4 and bh >= rh - 4:
                        continue

                    rect = cv2.minAreaRect(contour)
                    (rw_rect, rh_rect) = rect[1]
                    if rw_rect > 0 and rh_rect > 0:
                        box_area = rw_rect * rh_rect
                        fill_ratio = area / (box_area + 1e-5)
                        # Real ID cards fill most of their bounding rectangle
                        if fill_ratio >= 0.60:
                            aspect_ratio = max(rw_rect, rh_rect) / min(rw_rect, rh_rect)
                            if 1.0 <= aspect_ratio <= 2.5:
                                if aspect_ratio < 1.15 and (orig_area / total_area) < 0.25:
                                    continue
                                pts = cv2.boxPoints(rect).astype("float32") / scale
                                warped = four_point_transform(image, pts)
                                return warped, True

        return image, False

