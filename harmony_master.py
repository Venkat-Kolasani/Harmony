import cv2
from ultralytics import YOLO
import pygame
import math
import os
import time
import csv
import gc
import logging
import traceback
import numpy as np
from datetime import datetime
from color_segmentation import ColorSegmentationDetector

try:
    import torch
except Exception:
    torch = None

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
litter_model = YOLO('runs/detect/harmony_trash_v3/weights/last.pt')
coco_model   = YOLO('yolov8n.pt')

COCO_EXCLUSION_IDS = {0, 24, 26, 28, 56, 57, 58, 60, 63, 72, 74}
COCO_NAMES         = coco_model.names

# =============================================================================
# RUNTIME CONFIG / LOGGING
# =============================================================================
RUNTIME_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
DEBUG_LOG  = f"harmony_debug_{RUNTIME_TS}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(DEBUG_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("harmony")

CAMERA_INDEX            = int(os.getenv("HARMONY_CAMERA_INDEX", "0"))
FRAME_FAIL_RETRY_SLEEP  = float(os.getenv("HARMONY_FRAME_RETRY_SLEEP", "0.08"))
RECONNECT_AFTER_FAILS   = int(os.getenv("HARMONY_RECONNECT_AFTER_FAILS", "12"))
RECONNECT_MAX_ATTEMPTS  = int(os.getenv("HARMONY_RECONNECT_MAX_ATTEMPTS", "30"))
MAINTENANCE_EVERY_FRAMES = int(os.getenv("HARMONY_MAINTENANCE_EVERY_FRAMES", "150"))
SIMULATE_CAMERA_DROPS   = os.getenv("HARMONY_SIMULATE_CAMERA_DROPS", "0") == "1"
SIM_DROP_EVERY          = max(2, int(os.getenv("HARMONY_SIM_DROP_EVERY", "75")))

def _init_audio():
    try:
        pygame.mixer.init()
        logger.info("Audio initialized.")
        return True
    except Exception:
        logger.exception("Audio initialization failed; continuing without audio.")
        return False

def _open_camera(camera_idx=0, attempts=5, delay=0.35):
    for i in range(1, attempts + 1):
        cap_obj = cv2.VideoCapture(camera_idx)
        try:
            cap_obj.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if cap_obj is not None and cap_obj.isOpened():
            logger.info("Camera initialized (index=%s) on attempt %s/%s.", camera_idx, i, attempts)
            return cap_obj
        logger.warning("Camera open failed (index=%s), attempt %s/%s.", camera_idx, i, attempts)
        try:
            cap_obj.release()
        except Exception:
            pass
        time.sleep(delay)
    logger.error("Unable to initialize camera after %s attempts.", attempts)
    return None

audio_enabled = _init_audio()
cap = _open_camera(CAMERA_INDEX, attempts=8)
color_detector = ColorSegmentationDetector()

# =============================================================================
# DETECTION LOG
# =============================================================================
LOG_FILE        = f"harmony_log_{RUNTIME_TS}.csv"
_log_fh         = open(LOG_FILE, 'w', newline='')
_log_writer     = csv.writer(_log_fh)
_log_writer.writerow(["timestamp","frame","label","cx","cy","x1","y1","x2","y2","source"])

def log_detection(frame_num, label, x1, y1, x2, y2, source):
    cx, cy = (x1+x2)//2, (y1+y2)//2
    _log_writer.writerow([datetime.now().strftime('%H:%M:%S.%f')[:-3],
                          frame_num, label, cx, cy, x1, y1, x2, y2, source])
    _log_fh.flush()

# =============================================================================
# HEATMAP
# =============================================================================
_heatmap = None

def update_heatmap(detections, frame_shape):
    global _heatmap
    if _heatmap is None:
        _heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
    for (x1, y1, x2, y2, _) in detections:
        cx = int(np.clip((x1+x2)//2, 0, frame_shape[1]-1))
        cy = int(np.clip((y1+y2)//2, 0, frame_shape[0]-1))
        r  = 20
        _heatmap[max(0,cy-r):min(frame_shape[0],cy+r),
                 max(0,cx-r):min(frame_shape[1],cx+r)] += 1.0

def get_heatmap_overlay(h, w):
    if _heatmap is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    norm    = cv2.normalize(_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.resize(coloured, (w, h))

# =============================================================================
# SESSION STATS
# =============================================================================
class SessionStats:
    def __init__(self):
        self.start            = time.time()
        self.total_secondary  = 0
        self.total_yolo       = 0
        self.frames           = 0
        self.last_det         = None
        self.nudge_count      = 0
        self.cleanup_count    = 0
    def rec_sec(self, n):
        if n > 0: self.total_secondary += n; self.last_det = time.time()
    def rec_yolo(self, n): self.total_yolo += n
    def elapsed(self): return int(time.time() - self.start)
    def since_last(self):
        return int(time.time()-self.last_det) if self.last_det else None

stats = SessionStats()

# =============================================================================
# AUDIO INDICATOR
# =============================================================================
class AudioIndicator:
    DUR = 2.5
    def __init__(self): self._msg=""; self._t=0.0; self._on=False
    def trigger(self, msg): self._msg=msg; self._t=time.time(); self._on=True
    def draw(self, canvas, x, y):
        if not self._on: return
        e = time.time()-self._t
        if e > self.DUR: self._on=False; return
        a   = max(0.0, 1.0-(e/self.DUR))
        col = (int(0*a), int(220*a), int(255*a))
        cv2.putText(canvas, f"\u266a {self._msg}", (x,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

audio_ind = AudioIndicator()

# =============================================================================
# QUAD LAYOUT CONSTANTS
# =============================================================================
QW, QH = 640, 360          # Each quadrant size
FW, FH = QW*2, QH*2        # Full window 1280x720
HUD_H  = 40                # HUD strip height inside top-left quad

# =============================================================================
# SYSTEM CLASS
# =============================================================================
class HarmonyHOL:
    def __init__(self):
        self.nudge_played          = False
        self.thanked               = False
        self.interaction_started   = False
        self.waste_gone_frames     = 0
        self.required_frames       = 30
        self.person_absent_start   = 0
        self.abandon_timeout       = 5.0
        self.total_cleanups        = 0
        self.waste_verify_count    = 0
        self.threshold_nudge       = 350
        self.threshold_bin         = 150
        self.current_state         = "IDLE"
        self.person_was_near_waste = False
        self.secondary_count       = 0
        self.frame_num             = 0

    def perception(self, frame):
        self.frame_num += 1
        person = waste = bin_loc = None
        yolo_boxes = []
        excl_boxes = []

        # --- Litter model ---
        try:
            logger.debug("Frame %s: running litter model inference.", self.frame_num)
            lr = litter_model.predict(frame, conf=0.50, verbose=False)
        except Exception:
            logger.exception("Frame %s: litter model inference failed.", self.frame_num)
            raise
        n_yolo = 0
        for r in lr:
            for box in r.boxes:
                cls = int(box.cls[0])
                c   = box.xyxy[0]
                x1,y1,x2,y2 = int(c[0]),int(c[1]),int(c[2]),int(c[3])
                yolo_boxes.append([x1,y1,x2,y2]); n_yolo += 1
                if cls == 0:
                    waste = ((x1+x2)//2, (y1+y2)//2)
                    log_detection(self.frame_num,"litter_yolo",x1,y1,x2,y2,"YOLO")
        stats.rec_yolo(n_yolo)
        visual = lr[0].plot()

        # --- COCO model ---
        try:
            logger.debug("Frame %s: running COCO model inference.", self.frame_num)
            cr = coco_model.predict(frame, conf=0.45, verbose=False,
                                    classes=list(COCO_EXCLUSION_IDS))
        except Exception:
            logger.exception("Frame %s: COCO model inference failed.", self.frame_num)
            raise
        for r in cr:
            for box in r.boxes:
                cls = int(box.cls[0])
                c   = box.xyxy[0]
                x1,y1,x2,y2 = int(c[0]),int(c[1]),int(c[2]),int(c[3])
                lbl = COCO_NAMES.get(cls, str(cls))
                cv2.rectangle(visual,(x1,y1),(x2,y2),(255,255,0),2)
                cv2.putText(visual,lbl,(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,0),1)
                excl_boxes.append([x1,y1,x2,y2])
                if cls == 0: person = ((x1+x2)//2,(y1+y2)//2)

        # --- Color segmentation ---
        try:
            secs, bg_mask, hsv_mask = color_detector.detect(
                frame, yolo_litter_boxes=yolo_boxes, exclusion_boxes=excl_boxes)
        except Exception:
            logger.exception("Frame %s: color segmentation failed.", self.frame_num)
            raise
        self.secondary_count = len(secs)
        stats.rec_sec(self.secondary_count)
        stats.frames += 1

        visual = self._draw_weighted(visual, secs)
        for (x1,y1,x2,y2,lbl) in secs:
            log_detection(self.frame_num,lbl,x1,y1,x2,y2,"COLOR_SEC")
        update_heatmap(secs, frame.shape)

        if waste is None and secs:
            x1,y1,x2,y2,_ = secs[0]
            waste = ((x1+x2)//2,(y1+y2)//2)

        del lr, cr

        return person, waste, bin_loc, visual, bg_mask, hsv_mask

    def _draw_weighted(self, frame, detections):
        for (x1,y1,x2,y2,lbl) in detections:
            bx = ((x1+x2)//2) // color_detector.POSITION_TOLERANCE
            by = ((y1+y2)//2) // color_detector.POSITION_TOLERANCE
            held      = color_detector._duration_tracker.get((bx,by), 15)
            thickness = min(4, max(1, held // 15))
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),thickness)
            cv2.putText(frame,f"SEC({held}f)",(x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,0,255),1)
        return frame

    def spatial_analysis(self,p,w,b):
        dpw = math.sqrt((p[0]-w[0])**2+(p[1]-w[1])**2) if (p and w) else 999
        dwb = math.sqrt((w[0]-b[0])**2+(w[1]-b[1])**2) if (w and b) else 999
        return dpw, dwb

    def reset_system(self):
        self.nudge_played=self.thanked=self.interaction_started=False
        self.waste_gone_frames=self.person_absent_start=0
        self.person_was_near_waste=False; self.current_state="IDLE"

    def orchestrate(self,dpw,dwb,waste_v,person_v,bin_v):
        t = time.time()
        if waste_v: self.waste_verify_count=min(self.waste_verify_count+1,10)
        else:       self.waste_verify_count=max(self.waste_verify_count-1,0)
        real_waste = self.waste_verify_count >= 3

        if person_v and waste_v and dpw<200: self.person_was_near_waste=True

        if self.interaction_started and not person_v:
            if self.person_absent_start==0: self.person_absent_start=t
            elif (t-self.person_absent_start)>self.abandon_timeout:
                self.reset_system(); return
        else: self.person_absent_start=0

        if real_waste and person_v and dpw<self.threshold_nudge:
            if not self.interaction_started:
                self.interaction_started=True; self.current_state="(Sc) OBSERVING"

        if self.interaction_started and not self.nudge_played and not self.thanked:
            if not person_v or dpw>self.threshold_nudge:
                self.execute_action("nudge.mp3")
                self.nudge_played=True; self.current_state="(Sd) NUDGING"
                stats.nudge_count+=1; audio_ind.trigger("NUDGE SENT")

        if self.nudge_played and not self.thanked:
            at_bin     = (bin_v and waste_v and dwb<self.threshold_bin)
            picked_up  = (not waste_v and self.person_was_near_waste)
            if at_bin or picked_up:
                self.waste_gone_frames=min(self.waste_gone_frames+1,self.required_frames)
            else:
                if self.waste_gone_frames>0: self.waste_gone_frames-=1
            if self.waste_gone_frames>=self.required_frames:
                self.current_state="COMPLIANCE"; self.thanked=True

    def execute_action(self, fname):
        if os.path.exists(fname):
            try:
                if audio_enabled and (not pygame.mixer.music.get_busy()):
                    pygame.mixer.music.load(fname); pygame.mixer.music.play()
                    if fname=="thankyou.mp3": audio_ind.trigger("THANK YOU")
            except Exception:
                logger.exception("Audio action failed for file: %s", fname)


def _maintenance_tick(frame_idx):
    if frame_idx % MAINTENANCE_EVERY_FRAMES != 0:
        return
    try:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Frame %s: performed gc + torch.cuda.empty_cache().", frame_idx)
        else:
            logger.debug("Frame %s: performed gc.collect().", frame_idx)
    except Exception:
        logger.exception("Maintenance tick failed at frame %s.", frame_idx)


def _safe_release_camera(cam):
    try:
        if cam is not None:
            cam.release()
    except Exception:
        logger.exception("Camera release failed.")


def _cleanup(cam):
    _safe_release_camera(cam)
    try:
        cv2.destroyAllWindows()
    except Exception:
        logger.exception("cv2.destroyAllWindows failed.")
    try:
        _log_fh.close()
    except Exception:
        logger.exception("Detection CSV close failed.")


# =============================================================================
# DISPLAY BUILDERS
# =============================================================================
def _state_col(state):
    if "NUDG" in state: return (0,255,255)
    if "COMP" in state: return (0,255,0)
    return (180,180,180)

def quad_main(visual, state, sec_count):
    """Top-left: main feed + slim HUD strip."""
    feed = cv2.resize(visual, (QW, QH - HUD_H))
    hud  = np.zeros((HUD_H, QW, 3), dtype=np.uint8)
    cv2.rectangle(hud,(0,0),(QW,HUD_H),(20,20,20),-1)
    cv2.putText(hud,f"STATE: {state}",(10,26),
                cv2.FONT_HERSHEY_SIMPLEX,0.48,_state_col(state),1)
    sc = (255,0,255) if sec_count>0 else (80,80,80)
    cv2.putText(hud,f"SEC:{sec_count}",(QW-90,26),
                cv2.FONT_HERSHEY_SIMPLEX,0.50,sc,2)
    audio_ind.draw(hud, QW//2-80, 26)
    return np.vstack([hud, feed])

def quad_mask(mask, label):
    """Top-right / bottom-left: binary mask."""
    m3 = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR) if mask.ndim==2 else mask
    q  = cv2.resize(m3,(QW,QH))
    cv2.putText(q,label,(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.50,(160,160,160),1)
    return q

def quad_stats(state,p,w,b,wf,mf,cleanups,thanked):
    """Bottom-right: stats panel with heatmap background."""
    panel = np.zeros((QH,QW,3),dtype=np.uint8)
    hm    = get_heatmap_overlay(QH,QW)
    nz    = np.any(hm>10,axis=2)
    panel[nz] = (hm[nz]*0.5).astype(np.uint8)

    # Header bar
    cv2.rectangle(panel,(0,0),(QW,28),(25,25,25),-1)
    cv2.putText(panel,"STATS  +  LITTER HEATMAP",(8,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,220,255),1)

    el = stats.elapsed()
    rows = [
        (f"Session time     : {el//60:02d}:{el%60:02d}",          (200,200,200)),
        (f"YOLO detections  : {stats.total_yolo}",                  (0,255,0)),
        (f"SEC  detections  : {stats.total_secondary}",             (255,0,255)),
        (f"Frames processed : {stats.frames}",                      (160,160,160)),
        (f"Last SEC detect  : {f'{stats.since_last()}s ago' if stats.since_last() is not None else 'none yet'}",
                                                                     (160,160,160)),
        (f"Nudges sent      : {stats.nudge_count}",                 (0,200,255)),
        (f"Cleanups verified: {cleanups}",                          (0,255,120)),
    ]
    for i,(txt,col) in enumerate(rows):
        cv2.putText(panel,txt,(10,52+i*24),cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1)

    # State
    cv2.putText(panel,f"State: {state}",(10,232),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,_state_col(state),1)

    # Cleanup bar
    bx,by,bw,bh = 10,245,200,14
    cv2.rectangle(panel,(bx,by),(bx+bw,by+bh),(50,50,50),-1)
    prog = int((wf/mf)*bw)
    bc   = (0,255,0) if thanked else (0,200,255)
    if prog>0: cv2.rectangle(panel,(bx,by),(bx+prog,by+bh),bc,-1)
    cv2.putText(panel,"CLEANUP PROGRESS",(bx,by+28),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(130,130,130),1)

    # P/W/B dots
    for i,(nm,det) in enumerate([("P",p),("W",w),("B",b)]):
        ix,iy = 300+i*50, 258
        cv2.circle(panel,(ix,iy),14,(0,255,0) if det else (60,60,150),-1)
        cv2.putText(panel,nm,(ix-5,iy+5),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)

    # Legend
    cv2.putText(panel,"GREEN=litter  CYAN=COCO  MAGENTA=sec",
                (10,QH-10),cv2.FONT_HERSHEY_SIMPLEX,0.32,(120,120,120),1)
    return panel

def build_display(visual,bg,hsv,state,p,w,b,wf,mf,cleanups,thanked,sec_count):
    tl = quad_main(visual, state, sec_count)
    tr = quad_mask(bg,  "BG SUBTRACTION (MOG2)")
    bl = quad_mask(hsv, "HSV COLOUR MASK")
    br = quad_stats(state,p,w,b,wf,mf,cleanups,thanked)
    return np.vstack([np.hstack([tl,tr]), np.hstack([bl,br])])


# =============================================================================
# MAIN LOOP
# =============================================================================
hol = HarmonyHOL()
cv2.namedWindow("HARMONY HOL", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HARMONY HOL", FW, FH)

print("="*55)
print("  HARMONY HOL — Unified Display")
print("="*55)
print(f"  Log file : {LOG_FILE}")
print(f"  Window   : {FW}x{FH}  (2x2 quadrant layout)")
print()
print("  GREEN   = retrained litter model")
print("  CYAN    = COCO (person / bag / chair…)")
print("  MAGENTA = secondary color pipeline")
print("  Box thickness ∝ detection confidence (frames held)")
print("  Press Q to quit")
print("="*55)
print(f"  Debug log: {DEBUG_LOG}")

logger.info("Application startup complete.")
logger.info("Detection CSV: %s", LOG_FILE)
logger.info("Debug log: %s", DEBUG_LOG)
logger.info("Camera index=%s, reconnect_after_fails=%s, reconnect_max_attempts=%s",
            CAMERA_INDEX, RECONNECT_AFTER_FAILS, RECONNECT_MAX_ATTEMPTS)
if SIMULATE_CAMERA_DROPS:
    logger.warning("Camera drop simulation enabled. Dropping every %s frames.", SIM_DROP_EVERY)

frame_fail_streak = 0
reconnect_attempts = 0

try:
    while True:
        try:
            if cap is None or (not cap.isOpened()):
                reconnect_attempts += 1
                logger.warning("Camera is unavailable. Reconnect attempt %s/%s.",
                               reconnect_attempts, RECONNECT_MAX_ATTEMPTS)
                cap = _open_camera(CAMERA_INDEX, attempts=2, delay=0.5)
                if cap is None:
                    if reconnect_attempts >= RECONNECT_MAX_ATTEMPTS:
                        logger.error("Reached max reconnect attempts; pausing before retry cycle.")
                        reconnect_attempts = 0
                        time.sleep(1.0)
                    cv2.waitKey(1)
                    continue
                reconnect_attempts = 0

            ok, frame = cap.read()
            if SIMULATE_CAMERA_DROPS and (hol.frame_num > 0) and (hol.frame_num % SIM_DROP_EVERY == 0):
                ok, frame = False, None

            if (not ok) or (frame is None) or (getattr(frame, 'size', 0) == 0):
                frame_fail_streak += 1
                logger.warning("Frame capture failed (streak=%s).", frame_fail_streak)
                if frame_fail_streak >= RECONNECT_AFTER_FAILS:
                    logger.warning("Frame failure threshold reached. Releasing and reconnecting camera.")
                    _safe_release_camera(cap)
                    cap = None
                    frame_fail_streak = 0
                cv2.waitKey(1)
                time.sleep(FRAME_FAIL_RETRY_SLEEP)
                continue

            frame_fail_streak = 0

            try:
                p,w,b,vis,bg,hsv = hol.perception(frame)
                dpw, dwb = hol.spatial_analysis(p,w,b)
                hol.orchestrate(dpw,dwb, w is not None, p is not None, b is not None)

                is_comp = hol.thanked and hol.current_state=="COMPLIANCE"
                frame_out = build_display(vis,bg,hsv,
                                          hol.current_state,p,w,b,
                                          hol.waste_gone_frames,hol.required_frames,
                                          hol.total_cleanups, is_comp,
                                          hol.secondary_count)
                cv2.imshow("HARMONY HOL", frame_out)

                if is_comp:
                    cv2.waitKey(1)
                    hol.execute_action("thankyou.mp3")
                    hol.total_cleanups+=1; stats.cleanup_count+=1
                    time.sleep(2); hol.reset_system();

                _maintenance_tick(hol.frame_num)

                # Small delay / poll to keep UI responsive and avoid CPU saturation.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user.")
                    break

            except Exception as loop_err:
                logger.error("Per-frame processing error: %s", loop_err)
                logger.error(traceback.format_exc())
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user after frame exception.")
                    break
                time.sleep(0.01)
                continue

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Exiting cleanly.")
            break
        except Exception as outer_loop_err:
            logger.error("Main loop exception: %s", outer_loop_err)
            logger.error(traceback.format_exc())
            time.sleep(0.05)
            continue
finally:
    _cleanup(cap)

print(f"\nSession ended. Log → {LOG_FILE}")
print(f"Debug log        : {DEBUG_LOG}")
print(f"YOLO detections  : {stats.total_yolo}")
print(f"SEC  detections  : {stats.total_secondary}")
print(f"Frames processed : {stats.frames}")
print(f"Cleanups         : {stats.cleanup_count}")