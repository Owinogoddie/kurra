from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import OrderedDict  # Built-in, no installation needed

# ---------------- CONFIG FOR JETSON ORIN NANO -------------------
MODEL_PATH = "yolo11s.pt"
VIDEO_IN = "20.mp4"
VIDEO_OUT = "output_jetson.mp4"
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Frame dimensions to avoid distortion
TARGET_WIDTH = 1020
TARGET_HEIGHT = 570

LINE_Y = 400              # Counting line position (from your Colab)
MIN_CONF = 0.3            # Minimum confidence threshold (from your Colab)
MAX_DISAPPEAR = 30        # Tracker disappearance tolerance (from your Colab)

# ⚙️ DISPLAY SETTINGS - Change this to enable/disable real-time video
SHOW_DISPLAY = True       # Set to False if running headless (no monitor)
# ----------------------------------------------------------------

# Load YOLO model (will automatically use GPU on Jetson)
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

# ----------------- CENTROID TRACKER -----------------
class CentroidTracker:
    def __init__(self, max_disappear=30):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappear = max_disappear
    
    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappear:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects
        
        input_centroids = np.array(detections)
        
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.objects[self.next_id] = input_centroids[i]
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            D = np.linalg.norm(np.array(object_centroids)[:, None] - 
                             input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows, used_cols = set(), set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            
            for row in set(range(len(object_centroids))) - used_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappear:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            
            for col in set(range(len(input_centroids))) - used_cols:
                self.objects[self.next_id] = input_centroids[col]
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        
        return self.objects

# --------- Initialize video + tracker ----------
cap = cv2.VideoCapture(VIDEO_IN)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create output writer with TARGET dimensions
out = cv2.VideoWriter(VIDEO_OUT, fourcc, 30, (TARGET_WIDTH, TARGET_HEIGHT))

tracker = CentroidTracker(MAX_DISAPPEAR)

crossed_ids = set()
vehicle_count = 0
prev_time = time.time()

print(f"[INFO] Output resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
print(f"[INFO] Display mode: {'ENABLED' if SHOW_DISPLAY else 'DISABLED (headless)'}")
print("[INFO] Starting processing...")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Resize frame to target dimensions to avoid distortion
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    
    # Run YOLO inference (GPU automatically used on Jetson)
    results = model(frame, device=0, verbose=False)[0]
    
    detections = []
    class_names = model.names
    
    # Parse YOLO results
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        
        if cls_name not in VEHICLE_CLASSES:
            continue
        if box.conf[0] < MIN_CONF:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        detections.append((cx, cy))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, cls_name, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Update tracker
    objects = tracker.update(detections)
    
    # Count vehicles crossing the line
    for obj_id, (cx, cy) in objects.items():
        if cy > LINE_Y and obj_id not in crossed_ids:
            crossed_ids.add(obj_id)
            vehicle_count += 1
        
        cv2.putText(frame, f"ID {obj_id}", (cx, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)
    
    # Draw counting line
    cv2.line(frame, (0, LINE_Y), (TARGET_WIDTH, LINE_Y), (0, 0, 255), 2)
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    
    # Display info
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    
    # Write to output video
    out.write(frame)
    
    # ========== REAL-TIME DISPLAY CONTROL ==========
    # Change SHOW_DISPLAY at the top to enable/disable
    if SHOW_DISPLAY:
        cv2.imshow("Vehicle Detection Jetson Orin Nano", frame)
        
        # Press ESC to stop, or 'q' to quit
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or 'q'
            print("[INFO] Stopped by user")
            break
    # ===============================================
    
    # Progress indicator (every 30 frames)
    if frame_count % 30 == 0:
        print(f"[INFO] Processed {frame_count} frames | FPS: {fps:.1f} | Vehicles: {vehicle_count}")

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n[INFO] ========== PROCESSING COMPLETE ==========")
print(f"[INFO] Total frames processed: {frame_count}")
print(f"[INFO] Total vehicles counted: {vehicle_count}")
print(f"[INFO] Output saved to: {VIDEO_OUT}")
print(f"[INFO] ==========================================")