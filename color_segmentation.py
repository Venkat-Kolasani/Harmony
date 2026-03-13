import cv2
import numpy as np

# Secondary detector used to complement YOLO when small/low-confidence litter is missed.
class ColorSegmentationDetector:
    """
    Secondary litter detection module for Harmony HOL.
    Uses Background Subtraction (MOG2) and HSV Color Masking
    to detect litter that YOLOv8n fails to detect.

    Author: Kolasani Venkat
    Task: Group 3, Team C, Student 5 — Color Segmentation
    Theme: Digital Image Processing

    Pipeline:
        Stage 1 — MOG2 background subtraction (detects newly dropped litter by motion)
        Stage 2 — HSV colour masking (detects stationary litter by colour)
        Stage 3 — Contour filtering + morphological operations (removes noise)
        Stage 4 — IoU fusion gate with centroid-in-box check (removes YOLO duplicates)
        Stage 5 — Duration filter (suppresses transient false positives e.g. walking people)
    """

    def __init__(self):
        # --- Stage 1: MOG2 Background Subtractor ---
        # history=500: ~16 seconds of background memory at 30fps
        # varThreshold=50: conservative sensitivity — reduces cloud/shadow false triggers
        # detectShadows=True: marks shadow pixels grey (127) instead of white
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )

        # --- Stage 2: HSV Colour Ranges for Common Campus Litter ---
        # Each entry: (lower_bound, upper_bound) in HSV colour space.
        # HSV separates hue from brightness — same colour range works in sunlight and shade.
        self.hsv_ranges = {
            "red_wrapper": [
                (np.array([0,   100, 100]), np.array([10,  255, 255])),   # Lower red
                (np.array([160, 100, 100]), np.array([179, 255, 255]))    # Upper red (wraps in HSV)
            ],
            "yellow_wrapper": [
                (np.array([20, 100, 100]), np.array([35, 255, 255]))
            ],
            "orange_wrapper": [
                (np.array([10, 100, 100]), np.array([20, 255, 255]))
            ],
            "white_styrofoam": [
                (np.array([0, 0, 200]), np.array([179, 30, 245]))         # Cap V at 245, not 255
            ],
            "blue_plastic": [
                (np.array([100, 100, 100]), np.array([130, 255, 255]))
            ]
        }

        # --- Stage 3: Contour Filtering Parameters ---
        self.min_contour_area = 1500   # Below this = sensor noise or tiny reflection
        self.max_contour_area = 20000  # Above this = wall, person, large shadow (tightened from 30000)

        # --- Stage 5: Duration Filter ---
        # A secondary detection must persist in roughly the same position for
        # at least DURATION_THRESHOLD consecutive frames before being accepted.
        # Walking people clear the position in <0.5s. Dropped litter does not.
        self.DURATION_THRESHOLD = 15       # Frames required (~0.5s at 30fps)
        self.POSITION_TOLERANCE = 40       # Pixels — how close counts as "same position"
        self._duration_tracker = {}        # key: (cx, cy) bucket → frame count

        # --- Detection Output ---
        self.secondary_detections = []

    # =============================================
    # ROI MASK
    # =============================================
    def apply_roi_mask(self, mask, frame_height):
        """
        Blacks out the top SKY_FRACTION of the mask.
        Eliminates sky, building tops, and distant walls from consideration.
        """
        SKY_FRACTION = 0.30
        cutoff = int(frame_height * SKY_FRACTION)
        mask[:cutoff, :] = 0
        return mask

    # =============================================
    # STAGE 1: BACKGROUND SUBTRACTION
    # =============================================
    def apply_background_subtraction(self, frame):
        """
        Applies MOG2 to detect newly appeared foreground objects.
        Returns a binary mask — white = new foreground object.
        Shadow pixels (grey value 127) are removed before contour extraction.
        """
        fg_mask = self.bg_subtractor.apply(frame)

        # Threshold removes shadows (value 127) — keeps only solid white foreground
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)  # Remove small specks
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes

        # Remove sky region
        fg_mask = self.apply_roi_mask(fg_mask, frame.shape[0])

        return fg_mask

    # =============================================
    # STAGE 2: HSV COLOUR MASKING
    # =============================================
    def apply_hsv_masking(self, frame):
        """
        Converts frame to HSV and masks for known litter colours.
        Returns a combined binary mask of all detected colour regions.
        Hue channel is illumination-stable — same range works in sunlight and shade.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for colour_name, ranges in self.hsv_ranges.items():
            for (lower, upper) in ranges:
                mask = cv2.inRange(hsv_frame, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up the combined mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Remove sky region
        combined_mask = self.apply_roi_mask(combined_mask, frame.shape[0])

        return combined_mask

    # =============================================
    # STAGE 3: CONTOUR FILTERING
    # =============================================
    def extract_detections_from_mask(self, mask, label):
        """
        Finds contours in a binary mask and filters by area.
        Returns list of bounding boxes [x1, y1, x2, y2, label].
        min_contour_area rejects noise; max_contour_area rejects walls/people/bags.
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
    def fusion_gate(self, secondary_detections, yolo_litter_boxes, exclusion_boxes):
        """
        Removes secondary detections that duplicate YOLO litter detections,
        or that fall inside known non-litter regions (persons, large objects).

        Two suppression checks per secondary detection:
          1. Centroid-in-box: if center point of secondary detection falls inside
             any YOLO litter box → same object, suppress.
          2. IoU fallback: if IoU with any YOLO litter box > 0.30 → suppress.
          3. Exclusion zone: if center point falls inside any exclusion box
             (person, building etc from COCO) → suppress.

        yolo_litter_boxes : list of [x1,y1,x2,y2] from retrained litter model
        exclusion_boxes   : list of [x1,y1,x2,y2] from COCO (persons, buildings etc)
        """
        new_detections = []

        for det in secondary_detections:
            x1, y1, x2, y2, label = det
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            suppress = False

            # Check against YOLO litter boxes
            for ybox in yolo_litter_boxes:
                # Centroid-in-box check (robust for mismatched box sizes)
                if ybox[0] < cx < ybox[2] and ybox[1] < cy < ybox[3]:
                    suppress = True
                    break
                # IoU fallback (catches partial overlaps)
                if self.compute_iou((x1, y1, x2, y2), ybox) > 0.30:
                    suppress = True
                    break

            # Check against exclusion zones (persons, buildings from COCO)
            if not suppress:
                for ebox in exclusion_boxes:
                    if ebox[0] < cx < ebox[2] and ebox[1] < cy < ebox[3]:
                        suppress = True
                        break

            if not suppress:
                new_detections.append(det)

        return new_detections

    def compute_iou(self, boxA, boxB):
        """
        Computes Intersection over Union between two bounding boxes.
        Returns 0.0 when boxes do not overlap.
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
    # STAGE 5: DURATION FILTER
    # =============================================
    def apply_duration_filter(self, candidates):
        """
        Only accepts detections that have persisted in the same position for
        at least DURATION_THRESHOLD consecutive frames.

        Mechanism: center points are bucketed to a grid (POSITION_TOLERANCE px).
        Each bucket accumulates a frame count. Buckets that reach the threshold
        are accepted. Buckets not seen this frame are decremented and removed.

        Walking people: clear position in <15 frames → never accepted.
        Dropped litter: stays → accumulates → accepted after ~0.5s.
        """
        # Build set of active buckets this frame
        active_buckets = set()
        for (x1, y1, x2, y2, label) in candidates:
            cx = ((x1 + x2) // 2) // self.POSITION_TOLERANCE
            cy = ((y1 + y2) // 2) // self.POSITION_TOLERANCE
            active_buckets.add((cx, cy))
            self._duration_tracker[(cx, cy)] = self._duration_tracker.get((cx, cy), 0) + 1

        # Decrement and clean up buckets not seen this frame
        for bucket in list(self._duration_tracker.keys()):
            if bucket not in active_buckets:
                self._duration_tracker[bucket] -= 2  # Decay faster than accumulation
                if self._duration_tracker[bucket] <= 0:
                    del self._duration_tracker[bucket]

        # Accept only candidates whose bucket has reached the threshold
        accepted = []
        for (x1, y1, x2, y2, label) in candidates:
            cx = ((x1 + x2) // 2) // self.POSITION_TOLERANCE
            cy = ((y1 + y2) // 2) // self.POSITION_TOLERANCE
            if self._duration_tracker.get((cx, cy), 0) >= self.DURATION_THRESHOLD:
                accepted.append((x1, y1, x2, y2, label))

        return accepted

    # =============================================
    # MAIN ENTRY POINT
    # =============================================
    def detect(self, frame, yolo_litter_boxes=[], exclusion_boxes=[]):
        """
        Full pipeline: runs all stages and returns confirmed new detections.
        Call once per frame from the main Harmony loop.

        Parameters
        ----------
        frame             : BGR frame from camera
        yolo_litter_boxes : YOLO retrained model detections [x1,y1,x2,y2]
        exclusion_boxes   : COCO model detections to suppress (persons, buildings)

        Returns
        -------
        confirmed_detections : list of (x1,y1,x2,y2,label) — new litter only
        bg_mask              : MOG2 binary mask (for debug window)
        hsv_mask             : HSV colour mask (for debug window)
        """
        # Stage 1 — motion-based detection
        bg_mask = self.apply_background_subtraction(frame)
        bg_detections = self.extract_detections_from_mask(bg_mask, "litter_motion")

        # Stage 2 — colour-based detection
        hsv_mask = self.apply_hsv_masking(frame)
        hsv_detections = self.extract_detections_from_mask(hsv_mask, "litter_colour")

        # Merge both detection streams
        all_secondary = bg_detections + hsv_detections

        # Stage 4 — remove YOLO duplicates and exclusion zones
        filtered = self.fusion_gate(all_secondary, yolo_litter_boxes, exclusion_boxes)

        # Stage 5 — remove transient detections (walking people, flickering)
        confirmed = self.apply_duration_filter(filtered)

        self.secondary_detections = confirmed
        return confirmed, bg_mask, hsv_mask

    def draw_detections(self, frame, detections):
        """
        Draws magenta bounding boxes for confirmed secondary detections.
        Visually distinct from YOLO green boxes and COCO cyan boxes.
        """
        for (x1, y1, x2, y2, label) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"SEC: {label}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
        return frame