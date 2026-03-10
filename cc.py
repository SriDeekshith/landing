import cv2

# Open camera (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not detected")
    exit()

print("Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Raspberry Pi Camera Test", frame)

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
