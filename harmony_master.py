import cv2
from ultralytics import YOLO
import pygame
import math
import os
import time
import numpy as np
from color_segmentation import ColorSegmentationDetector

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
# Retrained litter model — detects campus litter (class 0 = litter only)
litter_model = YOLO('runs/detect/harmony_trash_v3/weights/last.pt')

# Original COCO model — detects 80 general classes (person, building, car etc.)
# Used for two purposes:
#   1. Visual display — shows what non-litter objects are present (cyan boxes)
#   2. Exclusion zones — prevents color module from flagging people as litter
# Ultralytics auto-downloads yolov8n.pt (~6MB) on first run if not present.
coco_model = YOLO('yolov8n.pt')

# COCO class IDs to use as exclusion zones for the secondary pipeline.
# These are objects the color module might falsely flag as litter.
# 0=person, 24=backpack, 26=handbag, 28=suitcase,
# 56=chair, 57=couch, 58=potted plant, 60=dining table,
# 63=laptop, 72=tv, 74=clock
COCO_EXCLUSION_IDS = {0, 24, 26, 28, 56, 57, 58, 60, 63, 72, 74}

# COCO class names for display labels
COCO_NAMES = coco_model.names   # dict: {0: 'person', 1: 'bicycle', ...}

pygame.mixer.init()
cap = cv2.VideoCapture(0)

color_detector = ColorSegmentationDetector()


class HarmonyHOL:
    def __init__(self):
        # Interaction/state flags
        self.nudge_played        = False
        self.thanked             = False
        self.interaction_started = False

        # Cleanup verification (temporal smoothing)
        self.waste_gone_frames = 0
        self.required_frames   = 30

        # User-abandonment timeout
        self.person_absent_start = 0
        self.abandon_timeout     = 5.0

        # Session counters
        self.total_cleanups      = 0
        self.waste_verify_count  = 0
        self.threshold_nudge     = 350
        self.threshold_bin       = 150

        self.waste_ids              = [0]    # Retrained model: class 0 = litter
        self.bin_id                 = None   # Bin detection disabled
        self.current_state          = "IDLE"
        self.person_was_near_waste  = False
        self.secondary_count        = 0

    def perception(self, frame):
        """
        Runs both models on the current frame.

        Retrained litter model  → green boxes  → drives state machine
        COCO model              → cyan boxes   → visual context + exclusion zones
        Color segmentation      → magenta boxes → catches what YOLO missed
        """
        person, waste, bin_loc = None, None, None
        yolo_litter_boxes = []   # For fusion gate — litter model detections
        exclusion_boxes   = []   # For fusion gate — COCO non-litter detections

        # ------------------------------------------------------------------
        # 1. Run retrained LITTER model
        #    conf=0.50 — raised from 0.40 to cut weak false detections on
        #    walls, floors and distant background objects.
        # ------------------------------------------------------------------
        litter_results = litter_model.predict(frame, conf=0.50, verbose=False)

        for r in litter_results:
            for box in r.boxes:
                cls    = int(box.cls[0])
                coords = box.xyxy[0]
                cx     = int((coords[0] + coords[2]) / 2)
                cy     = int((coords[1] + coords[3]) / 2)

                yolo_litter_boxes.append([
                    int(coords[0]), int(coords[1]),
                    int(coords[2]), int(coords[3])
                ])

                if cls == 0:   # litter
                    waste = (cx, cy)

        # Get the annotated frame from litter model (green boxes + labels)
        visual = litter_results[0].plot()

        # ------------------------------------------------------------------
        # 2. Run COCO model — conf=0.45, only classes we care about
        #    Shows original 80-class labels visually (person, building etc.)
        #    Also builds exclusion_boxes to protect the secondary pipeline.
        # ------------------------------------------------------------------
        coco_results = coco_model.predict(frame, conf=0.45, verbose=False,
                                          classes=list(COCO_EXCLUSION_IDS))

        for r in coco_results:
            for box in r.boxes:
                cls    = int(box.cls[0])
                coords = box.xyxy[0]
                x1, y1, x2, y2 = (int(coords[0]), int(coords[1]),
                                   int(coords[2]), int(coords[3]))
                label  = COCO_NAMES.get(cls, str(cls))

                # Draw COCO detection in CYAN on the visual frame
                cv2.rectangle(visual, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(visual, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 1)

                # Add to exclusion zones so color module ignores this region
                exclusion_boxes.append([x1, y1, x2, y2])

                # Use COCO person detection for state machine person tracking
                if cls == 0:   # person
                    person = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # ------------------------------------------------------------------
        # 3. Run Color Segmentation — secondary pipeline
        #    Passes both litter boxes (suppress duplicates) and
        #    exclusion boxes (suppress people/buildings)
        # ------------------------------------------------------------------
        secondary_detections, bg_mask, hsv_mask = color_detector.detect(
            frame,
            yolo_litter_boxes=yolo_litter_boxes,
            exclusion_boxes=exclusion_boxes
        )
        self.secondary_count = len(secondary_detections)

        # Draw magenta secondary detection boxes on the visual frame
        visual = color_detector.draw_detections(visual, secondary_detections)

        # If litter model missed waste but color module found something confirmed,
        # treat first secondary detection as waste proxy for state machine.
        if waste is None and len(secondary_detections) > 0:
            x1, y1, x2, y2, label = secondary_detections[0]
            waste = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        return person, waste, bin_loc, visual, bg_mask, hsv_mask

    def spatial_analysis(self, p, w, b):
        dist_pw = math.sqrt((p[0]-w[0])**2 + (p[1]-w[1])**2) if (p and w) else 999
        dist_wb = math.sqrt((w[0]-b[0])**2 + (w[1]-b[1])**2) if (w and b) else 999
        return dist_pw, dist_wb

    def reset_system(self):
        self.nudge_played            = False
        self.thanked                 = False
        self.interaction_started     = False
        self.waste_gone_frames       = 0
        self.person_absent_start     = 0
        self.person_was_near_waste   = False
        self.current_state           = "IDLE"

    def orchestrate(self, dist_pw, dist_wb, waste_v, person_v, bin_v):
        current_time = time.time()

        if waste_v: self.waste_verify_count = min(self.waste_verify_count + 1, 10)
        else:       self.waste_verify_count = max(self.waste_verify_count - 1, 0)
        is_real_waste = self.waste_verify_count >= 3

        if person_v and waste_v and dist_pw < 200:
            self.person_was_near_waste = True

        if self.interaction_started and not person_v:
            if self.person_absent_start == 0:
                self.person_absent_start = current_time
            elif (current_time - self.person_absent_start) > self.abandon_timeout:
                self.reset_system()
                return
        else:
            self.person_absent_start = 0

        if is_real_waste and person_v and dist_pw < self.threshold_nudge:
            if not self.interaction_started:
                self.interaction_started = True
                self.current_state = "(Sc) OBSERVING"

        if self.interaction_started and not self.nudge_played and not self.thanked:
            if not person_v or dist_pw > self.threshold_nudge:
                self.execute_action("nudge.mp3")
                self.nudge_played  = True
                self.current_state = "(Sd) NUDGING"

        if self.nudge_played and not self.thanked:
            at_bin           = (bin_v and waste_v and dist_wb < self.threshold_bin)
            actually_picked_up = (not waste_v and self.person_was_near_waste)

            if at_bin or actually_picked_up:
                self.waste_gone_frames = min(self.waste_gone_frames + 1, self.required_frames)
            else:
                if self.waste_gone_frames > 0:
                    self.waste_gone_frames -= 1

            if self.waste_gone_frames >= self.required_frames:
                self.current_state = "COMPLIANCE"
                self.thanked       = True

    def execute_action(self, file_name):
        if os.path.exists(file_name):
            try:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load(file_name)
                    pygame.mixer.music.play()
            except:
                pass


# =============================================================================
# UI — SLIM HUD INTERFACE
# =============================================================================
def create_slim_interface(main_frame, state, p, w, b,
                          frames, max_frames, count, is_thanked, sec_count):
    """
    Compact HUD above the main vision frame.
    Colour legend shown in HUD:
      GREEN  = retrained litter model detections
      CYAN   = COCO model detections (person, building etc.)
      MAGENTA = secondary color segmentation detections
    """
    h_orig, w_orig, _ = main_frame.shape
    hud_h  = 110
    canvas = np.zeros((h_orig + hud_h, w_orig, 3), dtype=np.uint8)
    canvas[hud_h:] = main_frame
    cv2.rectangle(canvas, (0, 0), (w_orig, hud_h), (25, 25, 25), -1)

    # Left — cleanup counter and state
    cv2.putText(canvas, f"CLEANS: {count}",  (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,  0.7, (0, 255, 0),    2)
    cv2.putText(canvas, f"STATE:  {state}",  (15, 75),
                cv2.FONT_HERSHEY_DUPLEX,   0.5, (0, 255, 255),  1)

    # Right — secondary detection counter (magenta when active)
    sec_color = (255, 0, 255) if sec_count > 0 else (100, 100, 100)
    cv2.putText(canvas, f"SEC: {sec_count}", (w_orig - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, sec_color, 2)

    # Colour legend — bottom right of HUD
    cv2.putText(canvas, "GREEN=litter  CYAN=COCO  MAGENTA=secondary",
                (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    # Status indicators — P / W / B circles
    labels = [("P", p), ("W", w), ("B", b)]
    for i, (name, detected) in enumerate(labels):
        x     = 220 + (i * 70)
        color = (0, 255, 0) if detected else (0, 0, 180)
        cv2.circle(canvas, (x, 50), 18, color, -1)
        cv2.putText(canvas, name, (x - 6, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Cleanup progress bar
    bar_x, bar_y, bar_w, bar_h = 430, 38, 180, 20
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    progress = int((frames / max_frames) * bar_w)

    if is_thanked:
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), -1)
    elif progress > 0:
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + progress, bar_y + bar_h), (0, 255, 255), -1)

    cv2.putText(canvas, "VERIFYING CLEANUP", (bar_x, bar_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return canvas


# =============================================================================
# MAIN LOOP
# =============================================================================
hol_system = HarmonyHOL()
cv2.namedWindow("HARMONY HOL",    cv2.WINDOW_NORMAL)
cv2.namedWindow("SEC: BG Mask",   cv2.WINDOW_NORMAL)
cv2.namedWindow("SEC: HSV Mask",  cv2.WINDOW_NORMAL)

print("Harmony HOL started.")
print("  GREEN boxes   = retrained litter model")
print("  CYAN boxes    = COCO model (person, building, chair etc.)")
print("  MAGENTA boxes = secondary color segmentation pipeline")
print("Press Q to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    (p_center, w_center, b_center,
     visual_feedback, bg_mask, hsv_mask) = hol_system.perception(frame)

    dist_pw, dist_wb = hol_system.spatial_analysis(p_center, w_center, b_center)
    hol_system.orchestrate(
        dist_pw, dist_wb,
        w_center is not None,
        p_center is not None,
        b_center is not None
    )

    if hol_system.thanked and hol_system.current_state == "COMPLIANCE":
        final_display = create_slim_interface(
            visual_feedback, "COMPLIANCE",
            p_center, w_center, b_center,
            hol_system.required_frames, hol_system.required_frames,
            hol_system.total_cleanups, True, hol_system.secondary_count
        )
        cv2.imshow("HARMONY HOL", final_display)
        cv2.waitKey(1)
        hol_system.execute_action("thankyou.mp3")
        hol_system.total_cleanups += 1
        time.sleep(2)
        hol_system.reset_system()
        continue

    final_display = create_slim_interface(
        visual_feedback, hol_system.current_state,
        p_center, w_center, b_center,
        hol_system.waste_gone_frames, hol_system.required_frames,
        hol_system.total_cleanups, False, hol_system.secondary_count
    )

    cv2.imshow("HARMONY HOL",   final_display)
    cv2.imshow("SEC: BG Mask",  bg_mask)
    cv2.imshow("SEC: HSV Mask", hsv_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()