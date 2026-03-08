"""
Enhanced calibrator using home plate + batter's box lines for more robust calibration.

From the MLB diagram:
- Home plate: 17" (0.43m) wide
- Gap to batter's box inside edge: 6" (0.15m) on each side
- Batter's box width: 4' (1.22m) each side

For training mats:
- WIDTH is standard (must be for proper training)
- DEPTH may vary from regulation

Coordinate system:
- Origin: center of home plate front edge
- X: positive toward first base (right when facing pitcher)
- Y: up
- Z: positive toward pitcher
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

# Measurements in meters (from MLB diagram)
PLATE_WIDTH = 0.4318       # 17 inches
PLATE_SIDE_DEPTH = 0.2159  # 8.5 inches (side section)
PLATE_APEX_DEPTH = 0.2159  # 8.5 inches (angled section to apex)
PLATE_TOTAL_DEPTH = PLATE_SIDE_DEPTH + PLATE_APEX_DEPTH  # 17 inches total

BOX_WIDTH = 1.2192         # 4 feet - the key constraint
BOX_GAP = 0.1524           # 6 inches from plate edge

# X-coordinates of batter's box edges (these are KNOWN from diagram)
# Right box (toward first base)
RIGHT_BOX_INSIDE_X = PLATE_WIDTH/2 + BOX_GAP      # 0.368m
RIGHT_BOX_OUTSIDE_X = RIGHT_BOX_INSIDE_X + BOX_WIDTH  # 1.587m

# Left box (toward third base)
LEFT_BOX_INSIDE_X = -PLATE_WIDTH/2 - BOX_GAP      # -0.368m
LEFT_BOX_OUTSIDE_X = LEFT_BOX_INSIDE_X - BOX_WIDTH  # -1.587m

# Home plate corners (5 points) - fully known 3D positions
PLATE_CORNERS_3D = np.array([
    [-PLATE_WIDTH/2, 0, 0],                           # 0: Front left
    [PLATE_WIDTH/2, 0, 0],                            # 1: Front right
    [PLATE_WIDTH/2, 0, -PLATE_SIDE_DEPTH],            # 2: Back right
    [0, 0, -PLATE_TOTAL_DEPTH],                       # 3: Apex (back point)
    [-PLATE_WIDTH/2, 0, -PLATE_SIDE_DEPTH],           # 4: Back left
], dtype=np.float32)


@dataclass
class WhiteLine:
    """A detected white line with its properties."""
    endpoints: np.ndarray  # (2, 2) array of [x, y] pixel coordinates
    center: Tuple[float, float]
    length_px: float
    angle_deg: float

    @property
    def is_horizontal(self) -> bool:
        """Check if line is roughly horizontal (within 20 degrees)."""
        return abs(self.angle_deg) < 20 or abs(self.angle_deg - 180) < 20


@dataclass
class DetectionResult:
    """Result of detecting calibration points."""
    plate_corners: Optional[np.ndarray] = None  # (5, 2) or None
    white_lines: List[WhiteLine] = field(default_factory=list)

    # Matched line endpoints with known 3D positions
    line_points_2d: Optional[np.ndarray] = None  # (N, 2)
    line_points_3d: Optional[np.ndarray] = None  # (N, 3)

    @property
    def n_points(self) -> int:
        n = 0
        if self.plate_corners is not None:
            n += len(self.plate_corners)
        if self.line_points_2d is not None:
            n += len(self.line_points_2d)
        return n

    def get_matched_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all matched 2D and 3D points for calibration."""
        points_2d = []
        points_3d = []

        if self.plate_corners is not None:
            points_2d.append(self.plate_corners)
            points_3d.append(PLATE_CORNERS_3D)

        if self.line_points_2d is not None and self.line_points_3d is not None:
            points_2d.append(self.line_points_2d)
            points_3d.append(self.line_points_3d)

        if not points_2d:
            return np.array([]), np.array([])

        return np.vstack(points_2d), np.vstack(points_3d)


class BattersBoxCalibrator:
    """
    Camera calibrator using home plate and batter's box lines.

    Detects:
    - Home plate (white pentagon) - 5 corners with known 3D
    - White batter's box lines - uses known WIDTH to establish 3D
    """

    def __init__(self, fov: float = 70.0):
        self.fov = fov
        self.cameras: Dict[str, dict] = {}

    def detect_white_regions(self, image: np.ndarray) -> np.ndarray:
        """Create mask of white regions."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # White detection
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def detect_plate_corners(self, image: np.ndarray, mask: np.ndarray = None,
                              x_direction: np.ndarray = None,
                              view_type: str = None) -> Optional[np.ndarray]:
        """Detect home plate pentagon corners, with netting-robust fallback.

        Args:
            image: Input image
            mask: White region mask (optional)
            x_direction: Direction for edge orientation (optional)
            view_type: 'side' or 'back' for view-specific corner ordering
        """
        h, w = image.shape[:2]
        img_center_x = w // 2

        # Try standard detection first
        result = self._detect_plate_standard(image, mask, x_direction, view_type)

        if result is not None:
            # Validate result - plate should be near horizontal center in bottom quarter
            centroid = result.mean(axis=0)
            dist_from_center = abs(centroid[0] - img_center_x)
            in_bottom_quarter = centroid[1] > h * 0.75

            # Accept if near center (within 1/4 of image width) and in bottom
            if dist_from_center < w * 0.25 and in_bottom_quarter:
                return result

        # Fallback: netting-robust detection using red mat reference
        return self._detect_plate_through_netting(image, x_direction, view_type)

    def _detect_plate_standard(self, image: np.ndarray, mask: np.ndarray = None,
                                x_direction: np.ndarray = None,
                                view_type: str = None) -> Optional[np.ndarray]:
        """Standard plate detection for clear views."""
        h, w = image.shape[:2]
        img_center_x = w // 2

        if mask is None:
            mask = self.detect_white_regions(image)

        # Focus on lower 60% of image
        ground_mask = mask.copy()
        ground_mask[:int(h * 0.4), :] = 0

        contours, _ = cv2.findContours(ground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)

            min_area = (w * h) * 0.001
            max_area = (w * h) * 0.1
            if area < min_area or area > max_area:
                continue

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 5:
                corners = approx.reshape(-1, 2).astype(np.float32)

                rect = cv2.minAreaRect(corners)
                rect_w, rect_h = rect[1]
                if rect_w > 0 and rect_h > 0:
                    aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
                    if aspect > 5:
                        continue

                # Check for black edges (netting artifacts)
                # Sample pixels along edges - real plate edges should not be dark
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                has_black_edge = False
                for i in range(5):
                    p1 = corners[i].astype(int)
                    p2 = corners[(i + 1) % 5].astype(int)
                    mid = ((p1 + p2) // 2)
                    mid = np.clip(mid, [0, 0], [w-1, h-1])
                    v_value = hsv_img[mid[1], mid[0], 2]
                    if v_value < 80:  # Dark pixel = likely netting
                        has_black_edge = True
                        break
                if has_black_edge:
                    continue

                # Check pentagon symmetry - home plate is symmetric about its axis
                # Find the apex (smallest interior angle) and check symmetry
                angles = []
                for i in range(5):
                    p_prev = corners[(i - 1) % 5]
                    p_curr = corners[i]
                    p_next = corners[(i + 1) % 5]
                    v1 = p_prev - p_curr
                    v2 = p_next - p_curr
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                    angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
                apex_idx = int(np.argmin(angles))

                # The two sides from apex should be roughly equal length
                apex = corners[apex_idx]
                left_neighbor = corners[(apex_idx - 1) % 5]
                right_neighbor = corners[(apex_idx + 1) % 5]
                left_dist = np.linalg.norm(apex - left_neighbor)
                right_dist = np.linalg.norm(apex - right_neighbor)
                symmetry_ratio = min(left_dist, right_dist) / (max(left_dist, right_dist) + 1e-6)

                # Skip non-symmetric pentagons (ratio < 0.7)
                if symmetry_ratio < 0.7:
                    continue

                # Score: strongly prioritize near horizontal center, in bottom portion
                centroid = corners.mean(axis=0)
                dist_from_center = abs(centroid[0] - img_center_x)
                in_bottom_quarter = centroid[1] > h * 0.75

                # Weight distance much more heavily - plate should be near center
                center_score = 1.0 / (1 + (dist_from_center / 100) ** 2)
                score = area * center_score * symmetry_ratio  # Also favor symmetric pentagons
                if in_bottom_quarter:
                    score *= 2

                candidates.append((score, corners))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return self._order_pentagon_corners(candidates[0][1], x_direction, view_type)

    def _detect_plate_through_netting(self, image: np.ndarray,
                                       x_direction: np.ndarray = None,
                                       view_type: str = None) -> Optional[np.ndarray]:
        """Detect plate through netting using red mat as reference."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Find red mat
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        mat_contour = max(contours, key=cv2.contourArea)
        mat_rect = cv2.boundingRect(mat_contour)
        mat_center_x = mat_rect[0] + mat_rect[2] // 2

        # White detection - more permissive to see through netting
        lower_white = np.array([0, 0, 130])
        upper_white = np.array([180, 70, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Focus on mat area
        mat_mask = np.zeros_like(white_mask)
        cv2.drawContours(mat_mask, [mat_contour], -1, 255, -1)
        white_in_mat = white_mask & mat_mask

        # Large closing to bridge gaps caused by netting
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        white_closed = cv2.morphologyEx(white_in_mat, cv2.MORPH_CLOSE, close_kernel)
        white_closed = cv2.morphologyEx(white_closed, cv2.MORPH_OPEN,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        contours, _ = cv2.findContours(white_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find best pentagon - should be near horizontal center in bottom quarter
        best_plate = None
        best_score = -1
        img_center_x = w // 2

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300 or area > 15000:  # Filter out merged blobs
                continue

            hull = cv2.convexHull(cnt)
            rect = cv2.boundingRect(hull)
            cx = rect[0] + rect[2] // 2
            cy = rect[1] + rect[3] // 2

            # Must be in bottom quarter of image
            if cy < h * 0.75:
                continue

            # Distance from horizontal center of image (not mat)
            dist_from_center = abs(cx - img_center_x)

            for eps in [0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
                if len(approx) == 5:
                    # Score: strongly favor near center
                    center_score = 1.0 / (1 + (dist_from_center / 100) ** 2)
                    score = area * center_score

                    if score > best_score:
                        best_score = score
                        best_plate = approx.reshape(-1, 2).astype(np.float32)
                    break

        if best_plate is not None:
            return self._order_pentagon_corners(best_plate, x_direction, view_type)

        return None

    def detect_white_lines(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        min_length: int = 50
    ) -> List[WhiteLine]:
        """Detect white lines (potential batter's box edges)."""
        h, w = image.shape[:2]

        if mask is None:
            mask = self.detect_white_regions(image)

        # Focus on lower portion
        ground_mask = mask.copy()
        ground_mask[:int(h * 0.4), :] = 0

        # Find contours that are elongated (line-like)
        contours, _ = cv2.findContours(ground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lines = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue

            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            center = rect[0]
            size = rect[1]
            angle = rect[2]

            # Check if elongated (line-like)
            if size[0] > 0 and size[1] > 0:
                length = max(size)
                width = min(size)

                if length < min_length:
                    continue

                aspect = length / width
                if aspect > 3:  # Elongated enough to be a line
                    # Get endpoints
                    box = cv2.boxPoints(rect)

                    # Find the two endpoints (midpoints of short edges)
                    # Sort by distance to get pairs
                    dists = []
                    for i in range(4):
                        for j in range(i+1, 4):
                            d = np.linalg.norm(box[i] - box[j])
                            dists.append((d, i, j))
                    dists.sort()

                    # Short edges are the first two pairs
                    short1 = (dists[0][1], dists[0][2])
                    short2 = (dists[1][1], dists[1][2])

                    # Endpoints are midpoints of short edges
                    ep1 = (box[short1[0]] + box[short1[1]]) / 2
                    ep2 = (box[short2[0]] + box[short2[1]]) / 2

                    endpoints = np.array([ep1, ep2])

                    # Compute angle (0 = horizontal)
                    dx = ep2[0] - ep1[0]
                    dy = ep2[1] - ep1[1]
                    line_angle = np.degrees(np.arctan2(dy, dx))

                    lines.append(WhiteLine(
                        endpoints=endpoints,
                        center=center,
                        length_px=length,
                        angle_deg=line_angle
                    ))

        return lines

    def match_lines_to_box(
        self,
        lines: List[WhiteLine],
        plate_corners: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Match detected lines to batter's box positions.

        For indoor mats with full-width lines, we extract points at known
        X-coordinates (where batter's box edges would be).

        Returns:
            (points_2d, points_3d) for the matched line points
        """
        if plate_corners is None or len(lines) == 0:
            return None, None

        h, w = image_shape

        # Get plate reference
        plate_center = plate_corners.mean(axis=0)
        plate_width_px = np.linalg.norm(plate_corners[0] - plate_corners[1])
        scale = plate_width_px / PLATE_WIDTH

        # Known X-coordinates in pixels (relative to plate center)
        x_positions = {
            'right_inside': plate_center[0] + RIGHT_BOX_INSIDE_X * scale,
            'right_outside': plate_center[0] + RIGHT_BOX_OUTSIDE_X * scale,
            'left_inside': plate_center[0] + LEFT_BOX_INSIDE_X * scale,
            'left_outside': plate_center[0] + LEFT_BOX_OUTSIDE_X * scale,
        }

        # Filter to long horizontal lines (spanning significant width)
        min_width = BOX_WIDTH * scale  # At least one box width
        horizontal_lines = [
            l for l in lines
            if l.is_horizontal and abs(l.endpoints[1, 0] - l.endpoints[0, 0]) > min_width
        ]

        if not horizontal_lines:
            return None, None

        matched_2d = []
        matched_3d = []

        for line in horizontal_lines:
            # Get line extent
            left_x = min(line.endpoints[0, 0], line.endpoints[1, 0])
            right_x = max(line.endpoints[0, 0], line.endpoints[1, 0])
            line_y = line.center[1]

            # Estimate Z from vertical position relative to plate
            dy_pixels = line_y - plate_center[1]
            z_estimate = -dy_pixels / scale * 0.3  # Rough depth estimate

            # For each known X position that falls within this line's span
            for name, x_px in x_positions.items():
                if left_x <= x_px <= right_x:
                    # Interpolate Y position along the line
                    # (for horizontal lines, Y is roughly constant)
                    t = (x_px - line.endpoints[0, 0]) / (line.endpoints[1, 0] - line.endpoints[0, 0] + 1e-6)
                    y_px = line.endpoints[0, 1] + t * (line.endpoints[1, 1] - line.endpoints[0, 1])

                    # Get 3D X coordinate
                    if 'right_inside' in name:
                        x_3d = RIGHT_BOX_INSIDE_X
                    elif 'right_outside' in name:
                        x_3d = RIGHT_BOX_OUTSIDE_X
                    elif 'left_inside' in name:
                        x_3d = LEFT_BOX_INSIDE_X
                    else:
                        x_3d = LEFT_BOX_OUTSIDE_X

                    matched_2d.append([x_px, y_px])
                    matched_3d.append([x_3d, 0, z_estimate])

        if not matched_2d:
            return None, None

        return np.array(matched_2d), np.array(matched_3d, dtype=np.float32)

    def detect_all(self, image: np.ndarray, view_type: str = None) -> DetectionResult:
        """Detect all calibration points in image.

        Args:
            image: Input image
            view_type: 'side' or 'back' - used for correct corner ordering.
                       If None, uses general algorithm.
        """
        mask = self.detect_white_regions(image)

        # Detect white lines FIRST to get field orientation
        white_lines = self.detect_white_lines(image, mask)

        # Get line direction (lines run parallel to Z axis, perpendicular to front edge)
        line_direction = self._get_line_direction(white_lines)

        # Detect plate with orientation info (line_direction = Z axis)
        plate_corners = self.detect_plate_corners(image, mask, line_direction, view_type)

        # Match lines to batter's box positions
        line_2d, line_3d = None, None
        if plate_corners is not None:
            line_2d, line_3d = self.match_lines_to_box(
                white_lines, plate_corners, image.shape[:2]
            )

        return DetectionResult(
            plate_corners=plate_corners,
            white_lines=white_lines,
            line_points_2d=line_2d,
            line_points_3d=line_3d
        )

    def _get_line_direction(self, lines: List[WhiteLine]) -> Optional[np.ndarray]:
        """
        Get the direction of white lines on the mat.

        The white lines run PARALLEL to the Z axis (toward/away from pitcher).
        The front edge of home plate is PERPENDICULAR to these lines.

        Returns the unit vector along the white lines (Z direction).
        """
        # Find the longest line (any orientation)
        if not lines:
            return None

        longest = max(lines, key=lambda l: l.length_px)

        ep1, ep2 = longest.endpoints
        direction = ep2 - ep1
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        return direction

    def _order_pentagon_corners(self, corners: np.ndarray, z_direction: np.ndarray = None,
                                  view_type: str = None) -> np.ndarray:
        """
        Order pentagon corners using view-specific geometry.

        Output order matches PLATE_CORNERS_3D:
        0: Front-left  (-X, Z=0)
        1: Front-right (+X, Z=0)
        2: Back-right  (+X, Z=-side_depth)
        3: Apex        (0, Z=-total_depth)
        4: Back-left   (-X, Z=-side_depth)

        Args:
            corners: Detected pentagon corners (5, 2)
            z_direction: Optional direction toward pitcher (for line-based ordering)
            view_type: 'side' or 'back' - determines ordering logic
        """
        n = len(corners)
        xs = corners[:, 0]
        ys = corners[:, 1]

        # Step 1: Find apex by smallest interior angle
        angles = []
        for i in range(n):
            p_prev = corners[(i - 1) % n]
            p_curr = corners[i]
            p_next = corners[(i + 1) % n]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)

        apex_idx = int(np.argmin(angles))

        # Use view-specific ordering if view_type is provided
        if view_type == 'side':
            # Side view: Apex is leftmost, front edge is on the right
            # FL has larger Y (lower in image), FR has smaller Y

            # Front corners = two rightmost points (excluding apex)
            right_sorted = np.argsort(xs)[::-1]
            front_candidates = [i for i in right_sorted if i != apex_idx][:2]

            if ys[front_candidates[0]] > ys[front_candidates[1]]:
                FL_idx, FR_idx = front_candidates[0], front_candidates[1]
            else:
                FL_idx, FR_idx = front_candidates[1], front_candidates[0]

            # Back corners = remaining (not apex, not front)
            back_candidates = [i for i in range(5) if i != apex_idx and i not in [FL_idx, FR_idx]]

            if ys[back_candidates[0]] > ys[back_candidates[1]]:
                BL_idx, BR_idx = back_candidates[0], back_candidates[1]
            else:
                BL_idx, BR_idx = back_candidates[1], back_candidates[0]

            return np.array([corners[FL_idx], corners[FR_idx], corners[BR_idx],
                             corners[apex_idx], corners[BL_idx]])

        elif view_type == 'back':
            # Back view: Apex is bottommost (largest Y), front edge is at top
            # FL has larger X (right side - appears left from camera behind)
            # FR has smaller X (left side - appears right from camera behind)

            # Override apex detection for back view - use bottommost point
            apex_idx = int(np.argmax(ys))

            # Front corners = two topmost points (smallest Y)
            top_sorted = np.argsort(ys)
            front_candidates = [i for i in top_sorted if i != apex_idx][:2]

            if xs[front_candidates[0]] > xs[front_candidates[1]]:
                FL_idx, FR_idx = front_candidates[0], front_candidates[1]
            else:
                FL_idx, FR_idx = front_candidates[1], front_candidates[0]

            # Back corners = remaining
            back_candidates = [i for i in range(5) if i != apex_idx and i not in [FL_idx, FR_idx]]

            if xs[back_candidates[0]] > xs[back_candidates[1]]:
                BL_idx, BR_idx = back_candidates[0], back_candidates[1]
            else:
                BL_idx, BR_idx = back_candidates[1], back_candidates[0]

            return np.array([corners[FL_idx], corners[FR_idx], corners[BR_idx],
                             corners[apex_idx], corners[BL_idx]])

        # Fallback: original algorithm using z_direction or winding
        adj_to_apex = {(apex_idx - 1) % n, (apex_idx + 1) % n}

        edges = []
        for i in range(n):
            j = (i + 1) % n
            if i == apex_idx or j == apex_idx:
                continue
            edge_vec = corners[j] - corners[i]
            edge_vec_norm = edge_vec / (np.linalg.norm(edge_vec) + 1e-6)
            length = np.linalg.norm(edge_vec)

            if z_direction is not None:
                cos_angle = abs(np.dot(edge_vec_norm, z_direction))
                perp_score = 1 - cos_angle
            else:
                perp_score = 0

            edges.append((perp_score, length, i, j))

        if not edges:
            return corners

        edges.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, _, front_idx1, front_idx2 = edges[0]

        total = 0
        for i in range(n):
            p1 = corners[i]
            p2 = corners[(i + 1) % n]
            total += (p2[0] - p1[0]) * (p2[1] + p1[1])
        is_clockwise = total > 0

        step = (front_idx2 - front_idx1) % n
        if step == 1:
            if is_clockwise:
                front_left_idx, front_right_idx = front_idx1, front_idx2
            else:
                front_left_idx, front_right_idx = front_idx2, front_idx1
        else:
            if is_clockwise:
                front_left_idx, front_right_idx = front_idx2, front_idx1
            else:
                front_left_idx, front_right_idx = front_idx1, front_idx2

        back_corners = list(adj_to_apex)

        if (back_corners[0] - front_left_idx) % n == 1 or (front_left_idx - back_corners[0]) % n == 1:
            back_left_idx, back_right_idx = back_corners[0], back_corners[1]
        else:
            back_left_idx, back_right_idx = back_corners[1], back_corners[0]

        return np.array([
            corners[front_left_idx],
            corners[front_right_idx],
            corners[back_right_idx],
            corners[apex_idx],
            corners[back_left_idx],
        ])

    def calibrate_camera(
        self,
        image: np.ndarray,
        view_name: str,
        min_points: int = 5
    ) -> Tuple[bool, int]:
        """Calibrate camera using detected points."""
        detection = self.detect_all(image)

        if detection.n_points < min_points:
            return False, detection.n_points

        points_2d, points_3d = detection.get_matched_points()

        h, w = image.shape[:2]
        focal_length = w / (2 * np.tan(np.radians(self.fov / 2)))
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros(5)

        success, rvec, tvec = cv2.solvePnP(
            points_3d.astype(np.float64),
            points_2d.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return False, detection.n_points

        # Compute reprojection error
        projected, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        reproj_error = np.mean(np.linalg.norm(projected - points_2d, axis=1))

        # Store calibration
        R, _ = cv2.Rodrigues(rvec)
        camera_pos = -R.T @ tvec.flatten()

        self.cameras[view_name] = {
            'matrix': camera_matrix,
            'dist': dist_coeffs,
            'rvec': rvec,
            'tvec': tvec,
            'R': R,
            'position': camera_pos,
            'size': (w, h),
            'n_points': detection.n_points,
            'reproj_error': reproj_error,
            'detection': detection,
        }

        return True, detection.n_points

    def visualize_detection(self, image: np.ndarray, detection: DetectionResult) -> np.ndarray:
        """Draw detected points on image."""
        vis = image.copy()

        # Draw plate corners in green
        if detection.plate_corners is not None:
            pts = detection.plate_corners.astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            for i, pt in enumerate(pts):
                cv2.circle(vis, tuple(pt), 5, (0, 255, 0), -1)
                cv2.putText(vis, f"P{i}", tuple(pt + [5, -5]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw detected white lines
        for i, line in enumerate(detection.white_lines):
            color = (255, 255, 0) if line.is_horizontal else (128, 128, 128)
            p1 = tuple(line.endpoints[0].astype(int))
            p2 = tuple(line.endpoints[1].astype(int))
            cv2.line(vis, p1, p2, color, 2)
            cv2.circle(vis, p1, 4, color, -1)
            cv2.circle(vis, p2, 4, color, -1)

        # Draw matched line points in magenta
        if detection.line_points_2d is not None:
            for i, pt in enumerate(detection.line_points_2d):
                cv2.circle(vis, tuple(pt.astype(int)), 6, (255, 0, 255), -1)
                cv2.putText(vis, f"L{i}", tuple(pt.astype(int) + [5, -5]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Add text
        cv2.putText(vis, f"Points: {detection.n_points}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis, f"Lines: {len(detection.white_lines)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return vis


def test_detection(session_dir: Path):
    """Test detection on a session."""
    print("=" * 70)
    print("BATTER'S BOX CALIBRATION TEST")
    print("=" * 70)

    calibrator = BattersBoxCalibrator()

    for view in ['side', 'back']:
        video_path = session_dir / f'{view}.mp4'
        if not video_path.exists():
            continue

        print(f"\n{view.upper()} VIEW:")

        cap = cv2.VideoCapture(str(video_path))

        for frame_idx in [30, 60, 90]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            detection = calibrator.detect_all(frame)

            print(f"  Frame {frame_idx}:")
            print(f"    Plate: {detection.plate_corners is not None}")
            print(f"    Lines found: {len(detection.white_lines)}")
            print(f"    Horizontal lines: {sum(1 for l in detection.white_lines if l.is_horizontal)}")
            print(f"    Matched points: {len(detection.line_points_2d) if detection.line_points_2d is not None else 0}")
            print(f"    Total points: {detection.n_points}")

            # Save visualization
            vis = calibrator.visualize_detection(frame, detection)
            out_path = session_dir / f'detection_{view}_{frame_idx}.jpg'
            cv2.imwrite(str(out_path), vis)

        cap.release()

    # Try calibration
    print("\n" + "-" * 70)
    print("CALIBRATION TEST")
    print("-" * 70)

    for view in ['side', 'back']:
        video_path = session_dir / f'{view}.mp4'
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ret, frame = cap.read()
        cap.release()

        if ret:
            success, n_points = calibrator.calibrate_camera(frame, view)
            if success:
                cam = calibrator.cameras[view]
                print(f"  {view}: {n_points} points, error={cam['reproj_error']:.2f}px, pos={cam['position']}")
            else:
                print(f"  {view}: calibration failed ({n_points} points)")


def main():
    import sys

    session_dir = Path(__file__).parent.parent / 'training_data' / 'session_001'
    if len(sys.argv) > 1:
        session_dir = Path(sys.argv[1])

    if not session_dir.exists():
        print(f"Session not found: {session_dir}")
        return

    test_detection(session_dir)


if __name__ == '__main__':
    main()
