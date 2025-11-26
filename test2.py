import cv2
import pandas as pd
from ultralytics import YOLO
import os
import numpy as np

# Load YOLO 11 model
model = YOLO('yolo11s.pt')

# YOLO 11 uses the same class list as YOLOv8
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Video input path
video_path = '/content/20.mp4'

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    print("Please check the file path and make sure the video is uploaded correctly.")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Get original video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video opened successfully!")
print(f"Original dimensions: {original_width}x{original_height}")
print(f"Target dimensions: 1020x570 (will resize each frame)")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")

# Set target dimensions - 1020x570 as requested
target_width = 1020
target_height = 570

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = '/content/output_video_bytetrack.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

# Check if video writer opened successfully
if not out.isOpened():
    print("Error: Could not open video writer")
    cap.release()
    exit()

# Initialize counters and tracking dictionaries
count = 0
tracked_vehicles = {}  # Store vehicle info and counting status

# Separate counters for each vehicle type going out of town (red line)
counter_out_cars = []
counter_out_trucks = []
counter_out_buses = []
counter_out_motorcycles = []

# Separate counters for each vehicle type going to town (blue line)
counter_to_town_cars = []
counter_to_town_trucks = []
counter_to_town_buses = []
counter_to_town_motorcycles = []

print("Starting video processing with YOLO 11 + ByteTrack...")
print("Each frame will be resized from original size to 1020x570 before processing...")

# Function to check if point crosses a line (improved for diagonal lines)
def point_crosses_line(x, y, x1, y1, x2, y2, tolerance=20):
    """
    Check if point (x,y) is close to the line from (x1,y1) to (x2,y2)
    Uses perpendicular distance calculation for better accuracy
    """
    # Calculate perpendicular distance from point to line
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # Avoid division by zero
    denominator = (A**2 + B**2)**0.5
    if denominator == 0:
        return False

    distance = abs(A * x + B * y + C) / denominator

    # Check if point is within the x and y range of the line segment (with some buffer)
    min_x, max_x = min(x1, x2) - 10, max(x1, x2) + 10
    min_y, max_y = min(y1, y2) - 10, max(y1, y2) + 10

    return (distance <= tolerance and
            min_x <= x <= max_x and
            min_y <= y <= max_y)

# Function to get appropriate counter list
def get_counter_list(vehicle_class, direction):
    """Get the appropriate counter list based on vehicle type and direction"""
    if 'car' in vehicle_class:
        return counter_out_cars if direction == 'out' else counter_to_town_cars
    elif 'truck' in vehicle_class or 'lorry' in vehicle_class or 'van' in vehicle_class:
        return counter_out_trucks if direction == 'out' else counter_to_town_trucks
    elif 'bus' in vehicle_class:
        return counter_out_buses if direction == 'out' else counter_to_town_buses
    elif 'motorcycle' in vehicle_class:
        return counter_out_motorcycles if direction == 'out' else counter_to_town_motorcycles
    else:
        return counter_out_cars if direction == 'out' else counter_to_town_cars  # fallback

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame")
        break

    count += 1

    # RESIZE FRAME BEFORE PROCESSING
    frame = cv2.resize(frame, (target_width, target_height))

    # YOLO 11 prediction with ByteTrack tracking
    results = model.track(frame,
                         persist=True,              # Keep track IDs consistent across frames
                         tracker="bytetrack.yaml",  # Use ByteTrack algorithm
                         conf=0.4,                  # Detection confidence threshold
                         iou=0.7,                   # IoU threshold for NMS
                         verbose=False)

    # Process tracked results
    if len(results[0].boxes) > 0 and results[0].boxes.id is not None:
        boxes = results[0].boxes

        for i in range(len(boxes)):
            # Get detection info
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            confidence = boxes.conf[i].cpu().numpy()
            class_id = int(boxes.cls[i].cpu().numpy())
            track_id = int(boxes.id[i].cpu().numpy())

            # Get class name
            if class_id < len(class_list):
                class_name = class_list[class_id].lower()
            else:
                continue  # Skip if class_id is out of range

            # Filter for vehicles only (cars, trucks, buses, motorcycles)
            vehicle_types = ['car', 'truck', 'bus', 'motorcycle', 'lorry', 'van']
            if any(vehicle in class_name for vehicle in vehicle_types):
                cx = int((x1 + x2) / 2)  # Center x
                cy = int((y1 + y2) / 2)  # Center y

                # Initialize tracking info if new vehicle
                if track_id not in tracked_vehicles:
                    tracked_vehicles[track_id] = {
                        'class': class_name,
                        'counted_out': False,
                        'counted_to_town': False,
                        'last_position': (cx, cy)
                    }

                # Update vehicle class and position
                tracked_vehicles[track_id]['class'] = class_name
                tracked_vehicles[track_id]['last_position'] = (cx, cy)

                # Draw bounding box for all tracked vehicles
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow box
                cv2.putText(frame, f"{class_name}-{track_id}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Check for crossing OUT OF TOWN line (red line: 200,270 to 470,243)
                if point_crosses_line(cx, cy, 200, 270, 470, 243):
                    if not tracked_vehicles[track_id]['counted_out']:
                        # Mark as counted and add to appropriate counter
                        tracked_vehicles[track_id]['counted_out'] = True

                        counter_list = get_counter_list(class_name, 'out')
                        if track_id not in counter_list:
                            counter_list.append(track_id)

                        # Visual feedback
                        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)  # Red circle
                        cv2.putText(frame, f"OUT-{class_name}-{track_id}", (cx, cy-15),
                                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

                # Check for crossing TO TOWN line (blue line: 485,245 to 896,209)
                if point_crosses_line(cx, cy, 485, 245, 896, 209):
                    if not tracked_vehicles[track_id]['counted_to_town']:
                        # Mark as counted and add to appropriate counter
                        tracked_vehicles[track_id]['counted_to_town'] = True

                        counter_list = get_counter_list(class_name, 'to_town')
                        if track_id not in counter_list:
                            counter_list.append(track_id)

                        # Visual feedback
                        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)  # Blue circle
                        cv2.putText(frame, f"TO-TOWN-{class_name}-{track_id}", (cx, cy-15),
                                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)

    # Draw counting lines and labels
    red_color = (0, 0, 255)      # Red for out of town
    blue_color = (255, 0, 0)     # Blue for to town
    text_color = (255, 255, 255) # White text
    green_color = (0, 255, 0)    # Green

    # Draw OUT OF TOWN line (red): from (200,270) to (470,243)
    cv2.line(frame, (200, 270), (470, 243), red_color, 4)
    cv2.putText(frame, 'OUT OF TOWN', (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_color, 2)

    # Draw TO TOWN line (blue): from (485,245) to (896,209)
    cv2.line(frame, (485, 245), (896, 209), blue_color, 4)
    cv2.putText(frame, 'TO TOWN', (500, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue_color, 2)

    # Display counters - OUT OF TOWN (left side)
    y_pos = 30
    cv2.putText(frame, '=== OUT OF TOWN ===', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_color, 2)
    y_pos += 25
    cv2.putText(frame, f'Cars: {len(counter_out_cars)}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += 20
    cv2.putText(frame, f'Trucks: {len(counter_out_trucks)}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += 20
    cv2.putText(frame, f'Buses: {len(counter_out_buses)}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += 20
    cv2.putText(frame, f'Motorcycles: {len(counter_out_motorcycles)}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Display counters - TO TOWN (left side, below out of town)
    y_pos += 35
    cv2.putText(frame, '=== TO TOWN ===', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue_color, 2)
    y_pos += 25
    cv2.putText(frame, f'Cars: {len(counter_to_town_cars)}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += 20
    cv2.putText(frame, f'Trucks: {len(counter_to_town_trucks)}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += 20
    cv2.putText(frame, f'Buses: {len(counter_to_town_buses)}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += 20
    cv2.putText(frame, f'Motorcycles: {len(counter_to_town_motorcycles)}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Calculate and display totals (top right)
    total_out = len(counter_out_cars) + len(counter_out_trucks) + len(counter_out_buses) + len(counter_out_motorcycles)
    total_to_town = len(counter_to_town_cars) + len(counter_to_town_trucks) + len(counter_to_town_buses) + len(counter_to_town_motorcycles)

    cv2.putText(frame, f'TOTAL OUT: {total_out}', (target_width-250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color, 2)
    cv2.putText(frame, f'TOTAL TO TOWN: {total_to_town}', (target_width-250, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue_color, 2)

    # Add frame info with ByteTrack indicator
    cv2.putText(frame, f'Frame: {count}/{total_frames}', (target_width-250, target_height-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Show number of currently tracked vehicles
    current_tracked = len(results[0].boxes) if len(results[0].boxes) > 0 and results[0].boxes.id is not None else 0
    cv2.putText(frame, f'ByteTrack vehicles: {current_tracked}', (target_width-250, target_height-40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Add ByteTrack watermark
    cv2.putText(frame, 'ByteTrack Enabled', (target_width-180, target_height-60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Write the processed frame to output video
    out.write(frame)

    # Print progress every 100 frames
    if count % 100 == 0:
        current_vehicles = len(results[0].boxes) if len(results[0].boxes) > 0 else 0
        print(f"Processed {count}/{total_frames} frames... Current tracked vehicles: {current_vehicles}")

# Release video capture and writer
cap.release()
out.release()

print(f"\nVideo processing completed with ByteTrack!")
print(f"Total frames processed: {count}")
print(f"Video was resized from {original_width}x{original_height} to {target_width}x{target_height}")

# Check if output file was created and display final statistics
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path)
    print(f"Output video saved successfully: {output_path}")
    print(f"File size: {file_size / (1024*1024):.2f} MB")

    print("\n" + "="*50)
    print("FINAL VEHICLE COUNTING STATISTICS (ByteTrack)")
    print("="*50)

    print("\nVEHICLES GOING OUT OF TOWN:")
    print(f"  Cars: {len(counter_out_cars)}")
    print(f"  Trucks/Lorries/Vans: {len(counter_out_trucks)}")
    print(f"  Buses: {len(counter_out_buses)}")
    print(f"  Motorcycles: {len(counter_out_motorcycles)}")
    total_out_final = len(counter_out_cars) + len(counter_out_trucks) + len(counter_out_buses) + len(counter_out_motorcycles)
    print(f"  TOTAL OUT OF TOWN: {total_out_final}")

    print("\nVEHICLES GOING TO TOWN:")
    print(f"  Cars: {len(counter_to_town_cars)}")
    print(f"  Trucks/Lorries/Vans: {len(counter_to_town_trucks)}")
    print(f"  Buses: {len(counter_to_town_buses)}")
    print(f"  Motorcycles: {len(counter_to_town_motorcycles)}")
    total_to_town_final = len(counter_to_town_cars) + len(counter_to_town_trucks) + len(counter_to_town_buses) + len(counter_to_town_motorcycles)
    print(f"  TOTAL TO TOWN: {total_to_town_final}")

    print(f"\nGRAND TOTAL VEHICLES COUNTED: {total_out_final + total_to_town_final}")
    print(f"Tracking Algorithm: ByteTrack (Improved accuracy & ID consistency)")

    # Display unique track IDs for verification
    print(f"\nTotal unique vehicle IDs tracked: {len(tracked_vehicles)}")

    # Download the processed video (for Google Colab)
    try:
        from google.colab import files
        files.download(output_path)
        print("\nVideo download initiated!")
    except ImportError:
        print(f"\nVideo saved locally at: {output_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        print(f"Video saved locally at: {output_path}")
else:
    print("Error: Output video file was not created!")

print("\nProcessing complete with ByteTrack!")
print("Benefits of ByteTrack:")
print("- More stable tracking IDs")
print("- Better handling of occlusions")
print("- Reduced duplicate counting")
print("- Improved motion prediction")