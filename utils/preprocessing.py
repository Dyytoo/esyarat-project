import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        """
        Detect hands in the frame and return landmarks
        Returns:
            frame: Frame dengan landmark yang digambar
            landmarks_list: List of landmarks for each hand
            hand_type: List of hand types ('Left' or 'Right')
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Initialize lists
        landmarks_list = []
        hand_type = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand type (Left or Right)
                hand_type.append(results.multi_handedness[idx].classification[0].label)
                
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extract landmarks
                hand_landmarks_list = []
                for landmark in hand_landmarks.landmark:
                    hand_landmarks_list.append([landmark.x, landmark.y, landmark.z])
                
                landmarks_list.append(hand_landmarks_list)
                
                # Add text label for hand type
                h, w, c = frame.shape
                x, y = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                cv2.putText(frame, hand_type[-1], (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame, landmarks_list, hand_type

    def preprocess_landmarks(self, landmarks_list):
        """
        Preprocess landmarks for model input
        Args:
            landmarks_list: List of landmarks for each hand
        Returns:
            processed_landmarks: Preprocessed landmarks ready for model input
        """
        if not landmarks_list:
            return None
            
        processed_landmarks = []
        for landmarks in landmarks_list:
            # Convert to numpy array
            landmarks = np.array(landmarks)
            
            # Normalize coordinates
            landmarks = (landmarks - landmarks.min()) / (landmarks.max() - landmarks.min())
            
            # Reshape for model input (assuming 21 landmarks per hand)
            landmarks = landmarks.reshape(1, 21, 3)
            processed_landmarks.append(landmarks)
        
        return processed_landmarks 