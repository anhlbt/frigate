import cv2

# rtsp_url = "rtsp://admin:L2F284FE@192.168.1.86:554/cam/realmonitor?channel=1&subtype=1"
rtsp_url = "rtsp://admin:admin@anhlbt113.smartddns.tv:554/cam/realmonitor?channel=1&subtype=1"
cap = cv2.VideoCapture(rtsp_url)

if cap.isOpened():
    print("RTSP stream is accessible!")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame.")
            break
        cv2.imshow("RTSP Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print("Failed to access RTSP stream.")

cap.release()
cv2.destroyAllWindows()
