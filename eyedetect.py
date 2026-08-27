import cv2
import math
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector


# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# Face Mesh
face_detector = FaceMeshDetector(maxFaces=1)

# Hand Detector
hand_detector = HandDetector(
    detectionCon=0.5,
    maxHands=2
)


# -----------------------------
# Distance function
# -----------------------------
def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# -----------------------------
# Check whether a point
# is inside a rectangle
# -----------------------------
def point_inside_box(point, box):
    x, y = point

    x1, y1, x2, y2 = box

    return x1 <= x <= x2 and y1 <= y <= y2


# -----------------------------
# Main loop
# -----------------------------
while True:

    success, img = cap.read()

    if not success:
        print("Could not access webcam")
        break

    # Flip camera so it behaves like a mirror
    img = cv2.flip(img, 1)

    # -----------------------------
    # Face Mesh
    # -----------------------------
    img, faces = face_detector.findFaceMesh(
        img,
        draw=False
    )

    # -----------------------------
    # Hand Detection
    # -----------------------------
    hands, img = hand_detector.findHands(
        img,
        draw=False
    )

    status = "NO FACE"


    # =====================================================
    # FACE DETECTED
    # =====================================================

    if faces:

        face = faces[0]

        # -----------------------------
        # LEFT EYE
        # -----------------------------

        left_top = face[159]
        left_bottom = face[145]

        left_left = face[33]
        left_right = face[133]


        # -----------------------------
        # RIGHT EYE
        # -----------------------------

        right_top = face[386]
        right_bottom = face[374]

        right_left = face[362]
        right_right = face[263]


        # -----------------------------
        # Eye dimensions
        # -----------------------------

        left_vertical = distance(
            left_top,
            left_bottom
        )

        left_horizontal = distance(
            left_left,
            left_right
        )


        right_vertical = distance(
            right_top,
            right_bottom
        )

        right_horizontal = distance(
            right_left,
            right_right
        )


        # -----------------------------
        # Eye ratios
        # -----------------------------

        left_ratio = (
            left_vertical /
            left_horizontal
        )

        right_ratio = (
            right_vertical /
            right_horizontal
        )


        eye_ratio = (
            left_ratio +
            right_ratio
        ) / 2


        # =================================================
        # CREATE EYE REGIONS
        # =================================================

        # Left eye bounding box
        left_eye_x1 = min(
            left_left[0],
            left_right[0]
        )

        left_eye_x2 = max(
            left_left[0],
            left_right[0]
        )

        left_eye_y1 = min(
            left_top[1],
            left_bottom[1]
        )

        left_eye_y2 = max(
            left_top[1],
            left_bottom[1]
        )


        # Add some padding around eye
        padding = 30

        left_eye_box = (
            left_eye_x1 - padding,
            left_eye_y1 - padding,
            left_eye_x2 + padding,
            left_eye_y2 + padding
        )


        # Right eye bounding box
        right_eye_x1 = min(
            right_left[0],
            right_right[0]
        )

        right_eye_x2 = max(
            right_left[0],
            right_right[0]
        )

        right_eye_y1 = min(
            right_top[1],
            right_bottom[1]
        )

        right_eye_y2 = max(
            right_top[1],
            right_bottom[1]
        )


        right_eye_box = (
            right_eye_x1 - padding,
            right_eye_y1 - padding,
            right_eye_x2 + padding,
            right_eye_y2 + padding
        )


        # =================================================
        # CHECK IF HAND IS COVERING EYE
        # =================================================

        hand_blocking = False

        for hand in hands:

            # Hand bounding box
            x, y, w, h = hand["bbox"]

            hand_box = (
                x,
                y,
                x + w,
                y + h
            )


            # Check hand center
            hand_center = hand["center"]

            left_blocked = point_inside_box(
                hand_center,
                left_eye_box
            )

            right_blocked = point_inside_box(
                hand_center,
                right_eye_box
            )


            if left_blocked or right_blocked:
                hand_blocking = True


        # =================================================
        # DETERMINE EYE STATUS
        # =================================================

        if hand_blocking:

            status = "EYES BLOCKED"

        elif eye_ratio < 0.20:

            status = "EYES CLOSED"

        else:

            status = "EYES OPEN"


        # =================================================
        # DISPLAY STATUS
        # =================================================

        cv2.putText(
            img,
            status,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3
        )


        cv2.putText(
            img,
            f"Eye Ratio: {eye_ratio:.2f}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # -----------------------------
    # Show camera
    # -----------------------------

    cv2.imshow(
        "Eye Detection",
        img
    )


   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------

cap.release()
cv2.destroyAllWindows()