import cv2
import os

name = input("Enter your name: ").strip()
save_dir = f"dataset/{name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press C to capture image | Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Faces", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{save_dir}/{count}.jpg", gray)
        print(f"Saved image {count}")

    if key == ord('q') or count >= 20:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Face capture completed")
