import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Entry

import threading
from queue import Queue
from PIL import Image, ImageTk


class WindSpeedApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Wind Speed Detection App")

        # Video file path
        self.video_path = ""

        # Windmill radius
        self.windmill_radius = 56.0

        # Variables for displaying results
        self.angular_speed_label = tk.Label(master, text="Average Angular Speed:")
        self.angular_speed_label.pack()

        self.linear_speed_label = tk.Label(master, text="Average Linear Speed (Wind Speed):")
        self.linear_speed_label.pack()

        # Entry widget for entering windmill radius
        self.radius_label = tk.Label(master, text="Enter Windmill Radius:")
        self.radius_label.pack()

        self.radius_entry = Entry(master)
        self.radius_entry.pack()

        # Open File Button
        self.open_file_button = tk.Button(master, text="Open Video File", command=self.open_file)
        self.open_file_button.pack()

        # Start Detection Button
        self.start_detection_button = tk.Button(master, text="Start Detection", command=self.start_detection)
        self.start_detection_button.pack()

        # Variable to check if the thread is running
        self.thread_running = False

        # Queue for communication between threads
        self.queue = Queue()

        # Canvas for displaying video
        self.canvas = tk.Canvas(master)
        self.canvas.pack()

        # Variable to hold the current video frame
        self.current_frame = None

    def open_file(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.MOV")])

    def start_detection(self):
        if not self.video_path:
            print("Please select a video file.")
            return

        # Get windmill radius from the entry widget
        try:
            self.windmill_radius = float(self.radius_entry.get())
        except ValueError:
            print("Please enter a valid number for the windmill radius.")
            return

        # Check if the thread is already running
        if self.thread_running:
            print("Detection is already running.")
            return

        # Set the variable to indicate the thread is running
        self.thread_running = True

        # Create a new thread for video processing
        video_thread = threading.Thread(target=self.process_video)
        video_thread.start()

        # Periodically check if the thread has finished
        self.master.after(100, self.check_thread_status)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        prev_frame = None
        angular_speeds = []

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                diff = cv2.absdiff(gray_frame, prev_frame)
                angular_speed = np.sum(diff) / diff.size
                angular_speeds.append(angular_speed)

            prev_frame = gray_frame

            # Convert the frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to ImageTk format
            image = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=image)

            # Update the current frame variable
            self.current_frame = imgtk

            # Put results in the queue for the main thread
            self.queue.put((angular_speeds, imgtk))

            # Display the frame in the canvas
            self.display_frame(imgtk)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()

        # Put None in the queue to indicate no valid measurements
        self.queue.put(None)

        # Set the variable to indicate the thread has finished
        self.thread_running = False

    def check_thread_status(self):
        # Periodically check if the thread is still running
        if self.thread_running:
            self.master.after(100, self.check_thread_status)
        else:
            # Thread has finished, process results from the queue
            result = self.queue.get()
            self.process_results(result)

    def process_results(self, result):
        if result is not None:
            angular_speeds, _ = result

            if angular_speeds:
                # Calculate the average angular speed
                avg_angular_speed = np.mean(angular_speeds)

                # Calculate the linear speed (wind speed)
                linear_speed = avg_angular_speed * self.windmill_radius

                # Convert linear speed to km/h
                wind_speed = linear_speed / 3.6

                # Update the GUI labels from the main thread
                self.master.after(0, lambda: self.angular_speed_label.config(
                    text=f"Average Angular Speed: {avg_angular_speed}"))
                self.master.after(0, lambda: self.linear_speed_label.config(
                    text=f"Average Linear Speed (Wind Speed): {wind_speed:.2f} m/s"))
            else:
                # Handle the case when angular_speeds is empty
                self.master.after(0, lambda: self.angular_speed_label.config(
                    text="Average Angular Speed: N/A"))
                self.master.after(0, lambda: self.linear_speed_label.config(
                    text="Average Linear Speed (Wind Speed): N/A"))

        else:
            print("No valid measurements found.")

    def display_frame(self, imgtk):
        # Update the canvas with the current video frame
        self.canvas.config(width=imgtk.width(), height=imgtk.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)


def main():
    root = tk.Tk()
    WindSpeedApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
