import tkinter as tk
from tkinter import messagebox
import threading
import os

def run_script(script):
    def task():
        os.system(f"python {script}")
    threading.Thread(target=task, daemon=True).start()

def info(msg):
    messagebox.showinfo("Information", msg)

root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("460x420")
root.resizable(False, False)

# Title
tk.Label(
    root,
    text="Face Recognition Attendance System",
    font=("Arial", 16, "bold")
).pack(pady=20)

# Buttons
tk.Button(
    root, text="📸 Capture Face Dataset",
    width=30, height=2,
    command=lambda: [info("Camera will open.\nPress C to capture, Q to quit."),
                     run_script("capture_faces.py")]
).pack(pady=8)

tk.Button(
    root, text="🧠 Train Face Model",
    width=30, height=2,
    command=lambda: [info("Training will start.\nPlease wait."),
                     run_script("face_train.py")]
).pack(pady=8)

tk.Button(
    root, text="📝 Start Attendance",
    width=30, height=2,
    command=lambda: [info("Attendance started.\nPress Q to stop."),
                     run_script("face_attendance.py")]
).pack(pady=8)

# Footer
tk.Label(
    root,
    text="Attendance is saved in attendance.xlsx",
    fg="green",
    font=("Arial", 10)
).pack(pady=20)

tk.Label(
    root,
    text="Developed using Python & OpenCV",
    font=("Arial", 9),
    fg="gray"
).pack()

root.mainloop()
