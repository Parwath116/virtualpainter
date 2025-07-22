import cv2
import mediapipe as mp
import numpy as np
import time
import os


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

canvas = None
frame_shape = None


draw_color = (0, 255, 0) 
thickness = 5
prev_point = None
mirror_mode = False
eraser_mode = False
selected_color = "green"  


def load_color_images():
    colors = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'blank': (0, 0, 0)
    }
    color_images = {}
    for key, color in colors.items():
        file_path = f"{key}.jpg"
        if os.path.exists(file_path):
            img = cv2.imread(file_path)
            if img is not None and img.shape == (50, 50, 3):
                color_images[key] = img
            else:
                print(f"Warning: {file_path} is invalid, creating placeholder.")
                img = np.zeros((50, 50, 3), dtype=np.uint8)
                img[:] = color
                cv2.imwrite(file_path, img)
                color_images[key] = img
        else:
            print(f"Creating {file_path} as it does not exist.")
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            img[:] = color
            cv2.imwrite(file_path, img)
            color_images[key] = img
    return color_images

colors = load_color_images()


prev_time = 0

def calculate_fps():
    global prev_time
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    return fps

def process_frame(frame):
    global canvas, frame_shape
    if frame_shape is None:
        frame_shape = frame.shape
        canvas = np.zeros(frame_shape, dtype=np.uint8)
    

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    return frame_rgb, hand_results, faces

def draw_landmarks_and_faces(frame, hand_results, faces):

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
    return frame

def count_fingers(hand_landmarks):

    finger_tips = [4, 8, 12, 16, 20]
    count = 0
    for tip in finger_tips[1:]:  # Exclude thumb
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
            count += 1
    # Thumb check
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x:
        count += 1
    return count

def handle_gestures(hand_results, frame):
    global draw_color, thickness, prev_point, mirror_mode, eraser_mode, selected_color
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            finger_count = count_fingers(hand_landmarks)
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            

            if finger_count == 1:
                mirror_mode = False
                eraser_mode = False
                if 10 <= x <= 260 and 10 <= y <= 60:
                    if 10 <= x < 60:
                        draw_color = (0, 0, 255)
                        selected_color = "red"
                    elif 60 <= x < 110:
                        draw_color = (255, 0, 0)
                        selected_color = "blue"
                    elif 110 <= x < 160:
                        draw_color = (0, 255, 0)
                        selected_color = "green"
                    elif 160 <= x < 210:
                        draw_color = (0, 255, 255)
                        selected_color = "yellow"
                    elif 210 <= x <= 260:
                        draw_color = (0, 0, 0)
                        selected_color = "blank"
                prev_point = None
            

            elif finger_count == 2:
                mirror_mode = False
                eraser_mode = False
                if prev_point:
                    cv2.line(canvas, prev_point, (x, y), draw_color, thickness)
                prev_point = (x, y)
            

            elif finger_count == 3:
                mirror_mode = True
                eraser_mode = False
                if prev_point:
                    cv2.line(canvas, prev_point, (x, y), draw_color, thickness)
                    mirror_x = frame.shape[1] - x
                    mirror_prev_x = frame.shape[1] - prev_point[0]
                    cv2.line(canvas, (mirror_prev_x, prev_point[1]), (mirror_x, y), draw_color, thickness)
                prev_point = (x, y)

            elif finger_count == 4:
                mirror_mode = False
                eraser_mode = True
                if prev_point:
                    cv2.line(canvas, prev_point, (x, y), (0, 0, 0), thickness * 2)
                prev_point = (x, y)
            
            else:
                prev_point = None

            cv2.putText(frame, f"Fingers: {finger_count}", (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Index: ({x}, {y})", (10, frame.shape[0] - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def display_ui(frame):

    y_offset = 10
    x_offset = 10
    for i, (key, img) in enumerate(colors.items()):
        frame[y_offset:y_offset+50, x_offset:x_offset+50] = img
        if key == selected_color:
            cv2.rectangle(frame, (x_offset-2, y_offset-2), (x_offset+52, y_offset+52), 
                         (255, 255, 255), 2)
        x_offset += 50

    rules = [
        "1 Finger: Select Color",
        "2 Fingers: Draw",
        "3 Fingers: Mirror Draw",
        "4 Fingers: Eraser",
        "Select colors from top bar"
    ]
    for i, rule in enumerate(rules):
        cv2.putText(frame, rule, (10, 70 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    status = f"Color: {selected_color} | Thickness: {thickness}px {'(Eraser)' if eraser_mode else ''}{' (Mirror)' if mirror_mode else ''}"
    cv2.putText(frame, status, (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    global canvas
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_rgb, hand_results, faces = process_frame(frame)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            handle_gestures(hand_results, frame)

            frame = cv2.add(frame, canvas)

            frame = draw_landmarks_and_faces(frame, hand_results, faces)
            display_ui(frame)

            fps = calculate_fps()
            cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1] - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Virtual Painter', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if _name_ == "_main_":
    main()