"""
Silly Gesture Detector 🤪
Detects hand gestures and tongue sticking out for maximum fun!
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class SillyGestureDetector:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        try:
            self.lips_style = self.mp_drawing_styles.get_default_face_mesh_lips_style()
        except AttributeError:
            # Older MediaPipe versions only expose contour styles
            self.lips_style = getattr(
                self.mp_drawing_styles,
                "get_default_face_mesh_contours_style",
                lambda: self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )()
        
        # Initialize face mesh for tongue detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize hand detector
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Gesture state tracking
        self.tongue_out_history = deque(maxlen=5)
        self.gesture_history = deque(maxlen=5)
        self.last_silly_time = 0
        self.silly_messages = [
            "👅 TONGUE DETECTED! So silly!",
            "🤘 ROCK ON! You rebel!",
            "✌️ PEACE OUT! Groovy baby!",
            "👍 THUMBS UP! Looking good!",
            "✊ FIST BUMP! Power move!",
            "👋 WAVE HELLO! Hey there!",
            "🖖 LIVE LONG AND PROSPER!",
            "🤙 HANG LOOSE! Radical dude!"
        ]
        self.current_message = ""
        self.message_time = 0
        self.combo_count = 0
    
    @staticmethod
    def _landmark_to_np(landmark):
        """Convert a MediaPipe landmark to a NumPy array"""
        return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)
        
    def detect_tongue(self, face_landmarks, frame_shape):
        """Detect if tongue is sticking out"""
        h, w = frame_shape[:2]
        
        try:
            # Key landmarks for tongue detection
            # Upper lip: 13, Lower lip: 14
            # Mouth interior points: 78, 308 (left/right corners inside)
            # Tongue tip area: 17 (approximately)
            if len(face_landmarks.landmark) < 309:
                return False, 0.0

            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            mouth_left = face_landmarks.landmark[78]
            mouth_right = face_landmarks.landmark[308]
            
            # Calculate mouth opening relative to width for scale invariance
            mouth_opening = abs(upper_lip.y - lower_lip.y)
            mouth_width = np.linalg.norm(
                np.array([mouth_left.x, mouth_left.y]) -
                np.array([mouth_right.x, mouth_right.y])
            )
            if mouth_width < 1e-4:
                return False, mouth_opening
            mouth_ratio = mouth_opening / (mouth_width + 1e-6)
            
            # Convert to pixel distance to avoid tiny movements triggering detection
            mouth_opening_px = mouth_opening * h
            
            # Require both a large ratio and enough absolute movement
            current_detection = mouth_ratio > 0.30 and mouth_opening_px > 12
            self.tongue_out_history.append(current_detection)
            
            # Smooth detection over a few frames to avoid flicker
            is_tongue_out = self.tongue_out_history.count(True) >= 3
            
            return is_tongue_out, mouth_opening
        except Exception as err:
            print(f"[WARN] Tongue detection skipped: {err}")
            return False, 0.0
    
    def detect_hand_gesture(self, hand_landmarks, hand_label="Right"):
        """Detect various hand gestures"""
        wrist = self._landmark_to_np(hand_landmarks.landmark[0])
        
        def joint_angle(a, b, c):
            """Return the angle at point b (degrees) for the triangle a-b-c"""
            ba = a - b
            bc = c - b
            if np.linalg.norm(ba) < 1e-5 or np.linalg.norm(bc) < 1e-5:
                return 0.0
            cos_angle = np.dot(ba, bc) / ((np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        def is_finger_extended(tip_id, pip_id, mcp_id):
            tip = self._landmark_to_np(hand_landmarks.landmark[tip_id])
            pip = self._landmark_to_np(hand_landmarks.landmark[pip_id])
            mcp = self._landmark_to_np(hand_landmarks.landmark[mcp_id])
            
            angle = joint_angle(tip, pip, mcp)
            pointing_up = tip[1] < pip[1] - 0.01
            distance_check = np.linalg.norm(tip[:2] - wrist[:2]) > np.linalg.norm(mcp[:2] - wrist[:2]) + 0.01
            return angle > 150 and (pointing_up or distance_check)
        
        # Thumb needs special handling based on handedness
        thumb_tip = self._landmark_to_np(hand_landmarks.landmark[4])
        thumb_ip = self._landmark_to_np(hand_landmarks.landmark[3])
        thumb_mcp = self._landmark_to_np(hand_landmarks.landmark[2])
        thumb_cmc = self._landmark_to_np(hand_landmarks.landmark[1])
        
        thumb_angle = joint_angle(thumb_tip, thumb_ip, thumb_mcp)
        if hand_label == "Right":
            thumb_axis_test = thumb_tip[0] < thumb_ip[0] - 0.015
        else:
            thumb_axis_test = thumb_tip[0] > thumb_ip[0] + 0.015
        thumb_distance_test = np.linalg.norm(thumb_tip[:2] - wrist[:2]) > np.linalg.norm(thumb_cmc[:2] - wrist[:2]) + 0.015
        thumb_extended = thumb_angle > 150 and (thumb_axis_test or thumb_distance_test)
        
        # Count extended fingers
        fingers_up = [
            thumb_extended,
            is_finger_extended(8, 6, 5),    # Index
            is_finger_extended(12, 10, 9),  # Middle
            is_finger_extended(16, 14, 13), # Ring
            is_finger_extended(20, 18, 17)  # Pinky
        ]
        
        num_fingers = sum(fingers_up)
        
        thumb, index, middle, ring, pinky = fingers_up
        non_thumb_count = index + middle + ring + pinky
        
        # Detect specific gestures (prioritize more specific shapes first)
        if non_thumb_count == 0 and not thumb:
            return "FIST", "✊"
        if index and not middle and not ring and pinky:
            return "ROCK_ON", "🤘"
        if index and middle and not ring and not pinky:
            return "PEACE", "✌️"
        if not index and not middle and not ring and pinky and thumb:
            return "HANG_LOOSE", "🤙"
        if index and not middle and not ring and not pinky:
            return "POINTING", "☝️"
        if thumb and not index and not middle and not ring and not pinky:
            return "THUMBS_UP", "👍"
        if num_fingers == 5:
            return "OPEN_HAND", "🖐️"
        if non_thumb_count == 4:
            return "FOUR", "4️⃣"
        if non_thumb_count == 3:
            return "THREE", "3️⃣"
        if non_thumb_count == 2:
            return "TWO", "2️⃣"
        
        return "UNKNOWN", "❓"
    
    def draw_silly_effects(self, frame, gesture_name, emoji, position):
        """Draw fun effects on the frame"""
        x, y = position
        
        # Create a fun message
        message = f"{emoji} {gesture_name}"
        
        # Draw with style
        cv2.putText(frame, message, (x, y),
                   cv2.FONT_HERSHEY_DUPLEX, 1.5,
                   (255, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, message, (x, y),
                   cv2.FONT_HERSHEY_DUPLEX, 1.5,
                   (147, 20, 255), 2, cv2.LINE_AA)
    
    def process_frame(self, frame):
        """Process a single frame and detect gestures"""
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Process face mesh safely to avoid crashing on bad frames
        face_results = None
        try:
            face_results = self.face_mesh.process(rgb_frame)
        except Exception as err:
            print(f"[WARN] Face mesh processing skipped: {err}")
            self.tongue_out_history.clear()
        
        # Process hands
        hand_results = None
        try:
            hand_results = self.hands.process(rgb_frame)
        except Exception as err:
            print(f"[WARN] Hand processing skipped: {err}")
        
        detected_gestures = []
        
        # Check for tongue
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                is_tongue, mouth_opening = self.detect_tongue(face_landmarks, frame.shape)
                
                if is_tongue:
                    detected_gestures.append(("TONGUE", "👅"))
                    # Draw mouth landmarks for fun
                    self.mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        self.mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.lips_style
                    )
        
        # Check for hand gestures
        if hand_results and hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                hand_label = "Right"
                if hand_results.multi_handedness and len(hand_results.multi_handedness) > idx:
                    hand_label = hand_results.multi_handedness[idx].classification[0].label
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Detect gesture
                gesture_name, emoji = self.detect_hand_gesture(hand_landmarks, hand_label)
                detected_gestures.append((gesture_name, emoji))
                
                # Get hand position for text
                wrist = hand_landmarks.landmark[0]
                x, y = int(wrist.x * w), int(wrist.y * h - 50)
                
                # Draw gesture name
                self.draw_silly_effects(frame, gesture_name, emoji, (max(10, x-100), max(50, y)))
        
        # Display combo counter
        if len(detected_gestures) > 1:
            self.combo_count += 1
            combo_text = f"COMBO x{self.combo_count}!"
            cv2.putText(frame, combo_text, (w//2 - 200, 50),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2,
                       (0, 255, 255), 3, cv2.LINE_AA)
        else:
            self.combo_count = max(0, self.combo_count - 1)
        
        # Display instructions
        cv2.putText(frame, "Show hand gestures and stick out your tongue!", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display all detected gestures at top
        if detected_gestures:
            gesture_text = " + ".join([f"{emoji} {name}" for name, emoji in detected_gestures])
            cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.putText(frame, gesture_text, (10, 70),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0,
                       (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame
    
    def run(self):
        """Main loop to run the detector"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Silly Gesture Detector Started!")
        print("Show hand gestures!")
        print("Stick out your tongue!")
        print("Try combining them for combos")
        print("Press 'q' to quit")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                continue
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Display
            cv2.imshow('Silly Gesture Detector', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Thanks for being silly! Goodbye!")


if __name__ == "__main__":
    detector = SillyGestureDetector()
    detector.run()
