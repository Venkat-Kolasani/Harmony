import cv2
import numpy as np

class ColorSegmentationDetector:
    """
    Secondary litter detection module for Harmony HOL.
    Uses Background Subtraction (MOG2) and HSV Color Masking
    to detect litter that YOLOv8n fails to detect.
    
    Author: Kolasani Venkat
    Task: Group 3, Team C, Student 5 — Color Segmentation
    Theme: Digital Image Processing
    """

    def __init__(self):
        # --- Stage 1: MOG2 Background Subtractor ---
        # history=500: number of frames used to build background model
        # varThreshold=50: sensitivity — higher = less sensitive to small changes
        # detectShadows=True: shadows are marked grey instead of white (reduces false positives)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )

        # --- Stage 2: HSV Color Ranges for Common Campus Litter ---
        # Each entry: (lower_bound, upper_bound) in HSV
        # These ranges are tuned for outdoor Malaysian campus environment
        self.hsv_ranges = {
            "red_wrapper": [
                (np.array([0, 100, 100]),   np.array([10, 255, 255])),   # Lower red
                (np.array([160, 100, 100]), np.array([179, 255, 255]))    # Upper red (red wraps around in HSV)
            ],
            "yellow_wrapper": [
                (np.array([20, 100, 100]),  np.array([35, 255, 255]))
            ],
            "orange_wrapper": [
                (np.array([10, 100, 100]),  np.array([20, 255, 255]))
            ],
            "white_styrofoam": [
                (np.array([0, 0, 200]), np.array([179, 30, 245]))  # Cap V at 245 not 255
            ],
            "blue_plastic": [
                (np.array([100, 100, 100]), np.array([130, 255, 255]))
            ]
        }

        # --- Stage 3: Contour Filtering Parameters ---
        self.min_contour_area = 1500    # Minimum pixel area — smaller = noise, ignored
        self.max_contour_area = 30000  # Maximum pixel area — larger = person/bag, ignored

        # --- Detection Output ---
        self.secondary_detections = []  # List of bounding boxes found by this module

    # =============================================
    # STAGE 1: BACKGROUND SUBTRACTION
    # =============================================
    def apply_background_subtraction(self, frame):
        """
        Applies MOG2 to detect newly appeared foreground objects.
        Returns a binary mask where white = new foreground object.
        """
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (marked as grey value 127 by MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)   # Remove small specks
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes

        return fg_mask

    # =============================================
    # STAGE 2: HSV COLOR MASKING
    # =============================================
    def apply_hsv_masking(self, frame):
        """
        Converts frame to HSV and masks for known litter colours.
        Returns a combined binary mask of all detected colour regions.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for colour_name, ranges in self.hsv_ranges.items():
            for (lower, upper) in ranges:
                mask = cv2.inRange(hsv_frame, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up the combined mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        return combined_mask

    # =============================================
    # STAGE 3: CONTOUR FILTERING
    # =============================================
    def extract_detections_from_mask(self, mask, label):
        """
        Finds contours in a binary mask and filters by size.
        Returns list of bounding boxes [x1, y1, x2, y2, label].
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_contour_area <= area <= self.max_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append((x, y, x + w, y + h, label))

        return detections

    # =============================================
    # STAGE 4: FUSION GATE
    # =============================================
    def fusion_gate(self, secondary_detections, yolo_boxes):
        """
        Removes secondary detections that already overlap with YOLO detections.
        Only returns genuinely NEW detections that YOLO missed.
        
        yolo_boxes: list of [x1, y1, x2, y2] from YOLO results
        """
        new_detections = []

        for det in secondary_detections:
            x1, y1, x2, y2, label = det
            overlaps_with_yolo = False

            for ybox in yolo_boxes:
                if self.compute_iou((x1, y1, x2, y2), ybox) > 0.3:
                    overlaps_with_yolo = True
                    break

            if not overlaps_with_yolo:
                new_detections.append(det)

        return new_detections

    def compute_iou(self, boxA, boxB):
        """
        Computes Intersection over Union between two bounding boxes.
        IoU > 0.3 means boxes significantly overlap — detection is not new.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea)

    # =============================================
    # MAIN ENTRY POINT
    # =============================================
    def detect(self, frame, yolo_boxes=[]):
        """
        Full pipeline: runs all 4 stages and returns new detections.
        Call this once per frame from the main Harmony loop.
        """
        # Stage 1
        bg_mask = self.apply_background_subtraction(frame)
        bg_detections = self.extract_detections_from_mask(bg_mask, "litter_motion")

        # Stage 2
        hsv_mask = self.apply_hsv_masking(frame)
        hsv_detections = self.extract_detections_from_mask(hsv_mask, "litter_colour")

        # Combine both
        all_secondary = bg_detections + hsv_detections

        # Stage 4 — filter out what YOLO already caught
        new_detections = self.fusion_gate(all_secondary, yolo_boxes)

        self.secondary_detections = new_detections
        return new_detections, bg_mask, hsv_mask

    def draw_detections(self, frame, detections):
        """
        Draws bounding boxes for secondary detections on the frame.
        Uses distinct colour (magenta) so they're visually separate from YOLO (green).
        """
        for (x1, y1, x2, y2, label) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta box
            cv2.putText(frame, f"SEC: {label}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
        return frame