import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

SMILE_THRESHOLD = 0.35
MONKEY_POSE_THRESHOLD = 0.15 
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

try:
    smiling_emoji = cv2.imread("smile.jpg")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.jpg")
    monkey_thinking = cv2.imread("Monkey.jpg")
    monkey_finger_up = cv2.imread("Monkey2.jpg")

    if smiling_emoji is None:
        raise FileNotFoundError("smile.jpg not found")
    if straight_face_emoji is None:
        raise FileNotFoundError("plain.png not found")
    if hands_up_emoji is None:
        raise FileNotFoundError("air.jpg not found")
    if monkey_thinking is None:
        raise FileNotFoundError("Monkey.jpg not found")
    if monkey_finger_up is None:
        monkey_finger_up = cv2.imread("Monekey2.jpg")
        if monkey_finger_up is None:
             raise FileNotFoundError("Monkey2.jpg or Monekey2.jpg not found")

    smiling_emoji = cv2.resize(smiling_emoji, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji = cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE)
    monkey_thinking = cv2.resize(monkey_thinking, EMOJI_WINDOW_SIZE)
    monkey_finger_up = cv2.resize(monkey_finger_up, EMOJI_WINDOW_SIZE)
    
except Exception as e:
    print("Error loading emoji images!")
    print(f"Details: {e}")
    print("\nExpected files:")
    print("- smile.jpg (smiling face)")
    print("- plain.png (straight face)")
    print("- air.jpg (hands up)")
    print("- Monkey.jpg (thinking monkey)")
    print("- Monkey2.jpg (finger up monkey)")
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

print("Controls:")
print("  Press 'q' to quit")
print("  Raise hands above shoulders for hands up")
print("  Put finger near chin for thinking monkey (Monkey.jpg)")
print("  Point finger near face for finger up monkey (Monkey2.jpg)")
print("  Smile for smiling emoji")
print("  Straight face for neutral emoji")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "STRAIGHT_FACE"
        face_anchor = None

        # MONKEEE
        results_pose = pose.process(image_rgb)
        results_hands = hands.process(image_rgb)
        results_face = face_mesh.process(image_rgb)

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            if (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y):
                current_state = "HANDS_UP"
        
        if current_state != "HANDS_UP":
            if results_face.multi_face_landmarks:
                face_anchor = results_face.multi_face_landmarks[0].landmark[4]

            if results_hands.multi_hand_landmarks and face_anchor:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    distance = np.sqrt((finger_tip.x - face_anchor.x)**2 + (finger_tip.y - face_anchor.y)**2)
                    
                    if distance < MONKEY_POSE_THRESHOLD:
                        
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        
                        Y_DIFFERENCE_FOR_UP = 0.2
                        
                        if wrist.y - finger_tip.y > Y_DIFFERENCE_FOR_UP: 
                            current_state = "MONKEY_UP" # MONKEY UPPP
                        else:
                            current_state = "MONKEY_THINKING"
                        break

        if current_state == "STRAIGHT_FACE":
            if results_face is None:
                results_face = face_mesh.process(image_rgb) 
            
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    left_corner = face_landmarks.landmark[291]
                    right_corner = face_landmarks.landmark[61]
                    upper_lip = face_landmarks.landmark[13]
                    lower_lip = face_landmarks.landmark[14]

                    mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
                    mouth_height = ((lower_lip.x - upper_lip.x)**2 + (lower_lip.y - upper_lip.y)**2)**0.5
                    
                    if mouth_width > 0:
                        mouth_aspect_ratio = mouth_height / mouth_width
                        if mouth_aspect_ratio > SMILE_THRESHOLD:
                            current_state = "SMILING"
                        else:
                            current_state = "STRAIGHT_FACE"
        
        if current_state == "HANDS_UP":
            emoji_to_display = hands_up_emoji
            emoji_name = "🙌"
        elif current_state == "MONKEY_THINKING":
            emoji_to_display = monkey_thinking
            emoji_name = "🤔 MONKEY THINKING"
        elif current_state == "MONKEY_UP":
            emoji_to_display = monkey_finger_up
            emoji_name = "☝️ MONKEY UP! THAT'S RETARDED!"
        elif current_state == "SMILING":
            emoji_to_display = smiling_emoji
            emoji_name = "😊"
        elif current_state == "STRAIGHT_FACE":
            emoji_to_display = straight_face_emoji
            emoji_name = "😐"
        else:
            emoji_to_display = blank_emoji
            emoji_name = "❓"

        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        cv2.putText(camera_frame_resized, f'STATE: {current_state} {emoji_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Emoji Output', emoji_to_display)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()