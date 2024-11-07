import cv2

cap = cv2.VideoCapture('/dev/video2')

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Check if resolution is set correctly
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f'Resolution set to: {int(width)}x{int(height)}')

# Capture frame to check
ret, frame = cap.read()
if ret:
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
