import cv2  # Import OpenCV for camera feed and drawing
from ultralytics import YOLO  # Import YOLO for AI object detection
import pygame  # Import Pygame for playing the MP3 sounds
import math  # Import Math to calculate distances between coordinates
import os  # Import OS to check if sound files exist in the folder
import time  # Import Time to handle delays and timeouts
import numpy as np  # Import Numpy to create the UI background bar

# --- 1. INITIALIZATION ---
model = YOLO('runs/detect/harmony_trash_v3/weights/last.pt')
pygame.mixer.init()  # Initialize the audio player engine
cap = cv2.VideoCapture(0)  # Connect to the default computer webcam

class HarmonyHOL:
    def __init__(self):
        self.nudge_played = False  # Track if the "nudge" audio has played
        self.thanked = False  # Track if the "thank you" audio has played
        self.interaction_started = False  # Track if the AI is currently "watching" a person
        self.waste_gone_frames = 0  # Counter for how long the trash has been missing
        self.required_frames = 30  # Goal: Trash must be gone for 30 frames to fill the bar
        self.person_absent_start = 0  # Timer to track when the person leaves the camera view
        self.abandon_timeout = 5.0  # Reset system if person is gone for 5 seconds
        self.total_cleanups = 0  # Counter for successful cleanups performed
        self.waste_verify_count = 0  # Counter to confirm trash is real (prevents flickering)
        self.threshold_nudge = 350  # Distance limit: if person moves further, AI nudges
        self.threshold_bin = 150  # Distance limit: trash must be this close to bin
        self.waste_ids = [0]   # Retrained model uses class 0 for all litter
        self.bin_id = None     # Bin detection removed — not in retrained dataset
        self.current_state = "IDLE"  # The starting status of the system
        self.person_was_near_waste = False  # Did the person actually touch the trash?

    def perception(self, frame):
        results = model.predict(frame, conf=0.4, verbose=False)  # Run AI on current frame
        person, waste, bin_loc = None, None, None  # Reset object positions
        for r in results:  # Loop through all detected objects
            for box in r.boxes:  # Loop through each bounding box
                cls = int(box.cls[0])  # Get the ID of the object
                coords = box.xyxy[0]  # Get coordinates [x1, y1, x2, y2]
                center = (int((coords[0] + coords[2]) / 2), int((coords[1] + coords[3]) / 2))  # Calculate center point
                if cls == 0: waste = center  # If ID 26, it's the Bin
        return person, waste, bin_loc, results[0].plot()  # Return positions and the visual frame

    def spatial_analysis(self, p, w, b):
        # Calculate distance between Person and Waste
        dist_pw = math.sqrt((p[0]-w[0])**2 + (p[1]-w[1])**2) if (p and w) else 999
        # Calculate distance between Waste and Bin
        dist_wb = math.sqrt((w[0]-b[0])**2 + (w[1]-b[1])**2) if (w and b) else 999
        return dist_pw, dist_wb  # Return the two calculated distances

    def reset_system(self):
        self.nudge_played = False  # Reset nudge flag
        self.thanked = False  # Reset thanked flag
        self.interaction_started = False  # Reset interaction status
        self.waste_gone_frames = 0  # Reset the loading bar counter
        self.person_absent_start = 0  # Reset abandonment timer
        self.person_was_near_waste = False  # Reset proximity check
        self.current_state = "IDLE"  # Set state back to IDLE

    def orchestrate(self, dist_pw, dist_wb, waste_v, person_v, bin_v):
        current_time = time.time()  # Get the current system time
        if waste_v: self.waste_verify_count = min(self.waste_verify_count + 1, 10)  # Count up if trash seen
        else: self.waste_verify_count = max(self.waste_verify_count - 1, 0)  # Count down if trash missing
        is_real_waste = self.waste_verify_count >= 3  # Trash is "confirmed" after 3 frames

        if person_v and waste_v and dist_pw < 200:  # If person is very close to trash
            self.person_was_near_waste = True  # Mark them as the one who will clean it

        if self.interaction_started and not person_v:  # If system is active but person disappears
            if self.person_absent_start == 0: self.person_absent_start = current_time  # Start timer
            elif (current_time - self.person_absent_start) > self.abandon_timeout:  # If gone too long
                self.reset_system()  # Reset back to IDLE
                return
        else: self.person_absent_start = 0  # Stop timer if person reappears

        if is_real_waste and person_v and dist_pw < self.threshold_nudge:  # If trash and person are near
            if not self.interaction_started:  # If not already watching
                self.interaction_started = True  # Start the interaction
                self.current_state = "(Sc) OBSERVING"  # Update state to Observing

        if self.interaction_started and not self.nudge_played and not self.thanked:  # If watching but not nudged
            if not person_v or dist_pw > self.threshold_nudge:  # If person walks away
                self.execute_action("nudge.mp3")  # Play nudge audio
                self.nudge_played = True  # Mark nudge as done
                self.current_state = "(Sd) NUDGING"  # Update state to Nudging

        if self.nudge_played and not self.thanked:  # If nudged and waiting for cleanup
            at_bin = (bin_v and waste_v and dist_wb < self.threshold_bin)  # Trash is in the bin
            actually_picked_up = (not waste_v and self.person_was_near_waste)  # Trash was picked up
            
            if at_bin or actually_picked_up:  # If either cleanup condition is met
                self.waste_gone_frames = min(self.waste_gone_frames + 1, self.required_frames)  # Fill the bar
            else:
                if self.waste_gone_frames > 0: self.waste_gone_frames -= 1  # Empty the bar if trash returns
            
            if self.waste_gone_frames >= self.required_frames:  # If the bar is 100% full
                self.current_state = "COMPLIANCE"  # Update state to Success
                self.thanked = True  # Mark as finished (triggers Green Bar in UI)

    def execute_action(self, file_name):
        if os.path.exists(file_name):  # If the MP3 file exists in the folder
            try:
                if not pygame.mixer.music.get_busy():  # If no other sound is playing
                    pygame.mixer.music.load(file_name)  # Load the sound
                    pygame.mixer.music.play()  # Play the sound
            except: pass  # Skip if there is an audio error

# --- 2. UI INTERFACE ---
def create_slim_interface(main_frame, state, p, w, b, frames, max_frames, count, is_thanked):
    h_orig, w_orig, _ = main_frame.shape  # Get original camera size
    hud_h = 100  # Set height for the black top bar
    canvas = np.zeros((h_orig + hud_h, w_orig, 3), dtype=np.uint8)  # Create black canvas
    canvas[hud_h:] = main_frame  # Place camera feed below the black bar
    cv2.rectangle(canvas, (0, 0), (w_orig, hud_h), (25, 25, 25), -1)  # Draw dark background
    
    cv2.putText(canvas, f"CLEANS: {count}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Draw cleanup count
    cv2.putText(canvas, f"STATE: {state}", (15, 75), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)  # Draw current state

    labels = [("P", p), ("W", w), ("B", b)]  # List of sensor points
    for i, (name, detected) in enumerate(labels):  # Loop through sensors
        x = 220 + (i * 70)  # Calculate horizontal spacing
        color = (0, 255, 0) if detected else (0, 0, 180)  # Green if found, Red if not
        cv2.circle(canvas, (x, 45), 18, color, -1)  # Draw the sensor light
        cv2.putText(canvas, name, (x-6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Label the sensor

    bar_x, bar_y, bar_w, bar_h = 430, 35, 180, 20  # Set dimensions for the Verification Bar
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)  # Draw bar background
    
    progress = int((frames / max_frames) * bar_w)  # Calculate how much of the bar is full
    
    if is_thanked:  # If cleanup is finished
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), -1)  # Force Green
    elif progress > 0:  # If cleanup is in progress
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + progress, bar_y + bar_h), (0, 255, 255), -1)  # Draw Yellow
    
    cv2.putText(canvas, "VERIFYING CLEANUP", (bar_x, bar_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)  # Bar label
    return canvas  # Return the completed image

# --- 3. MAIN LOOP ---
hol_system = HarmonyHOL()  # Create the system instance
cv2.namedWindow("HARMONY HOL", cv2.WINDOW_NORMAL)  # Create the display window

while cap.isOpened():  # While the camera is working
    success, frame = cap.read()  # Read a single image from the camera
    if not success: break  # Stop if the camera fails
    
    p_center, w_center, b_center, visual_feedback = hol_system.perception(frame)  # Find objects
    dist_pw, dist_wb = hol_system.spatial_analysis(p_center, w_center, b_center)  # Measure distances
    
    hol_system.orchestrate(dist_pw, dist_wb, w_center is not None, p_center is not None, b_center is not None)  # Decision logic
    
    if hol_system.thanked and hol_system.current_state == "COMPLIANCE":  # Check if cleanup was just finished
        # Draw the frame one last time with the 100% Green Bar
        final_display = create_slim_interface(visual_feedback, "COMPLIANCE", p_center, w_center, b_center, 
                                            hol_system.required_frames, hol_system.required_frames, 
                                            hol_system.total_cleanups, True)
        cv2.imshow("HARMONY HOL", final_display)  # Show the Green Bar
        cv2.waitKey(1)  # Refresh window
        
        hol_system.execute_action("thankyou.mp3")  # Play the thank you sound
        hol_system.total_cleanups += 1  # Add to the global counter
        time.sleep(2)  # Pause for 2 seconds so the user can see the Green Bar
        hol_system.reset_system()  # Reset back to IDLE
        continue  # Skip to the next loop iteration

    # Create the standard interface (IDLE, Observing, or Nudging)
    final_display = create_slim_interface(
        visual_feedback, hol_system.current_state, p_center, w_center, b_center, 
        hol_system.waste_gone_frames, hol_system.required_frames, hol_system.total_cleanups, False
    )
    
    cv2.imshow("HARMONY HOL", final_display)  # Display the image
    if cv2.waitKey(1) & 0xFF == ord('q'): break  # Exit if 'q' is pressed

cap.release()  # Release the camera hardware
cv2.destroyAllWindows()  # Close all display windows