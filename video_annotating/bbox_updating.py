import tkinter as tk
from tkinter import Canvas
import cv2
import pandas as pd
from PIL import Image, ImageTk

class VideoAnnotationApp:
    def __init__(self, video_path, csv_path, resized_width=1440, resized_height=900):
        self.video_path = video_path
        self.csv_path = csv_path
        self.resized_width = resized_width
        self.resized_height = resized_height

        self.cap = cv2.VideoCapture(video_path)
        self.annotations = pd.read_csv(csv_path)

        # Calculate scaling factors
        self.x_scale = self.resized_width / int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.y_scale = self.resized_height / int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_annotations = self._load_annotations()
        self.bboxes = []
        self.selected_box = None
        self.moving = False
        self.resizing = False
        self.start_x, self.start_y = 0, 0
        self.current_frame = 0


        self._setup_tkinter_ui()

    def _load_annotations(self):
        annotations = {}
        
        for index, row in self.annotations.iterrows():
            frame_id = row['frame_id']            
            # Adjust bboxes to match resized frame
            bbox = (int(row['x1'] * self.x_scale), int(row['y1'] * self.y_scale),
                    int((row['x2'] - row['x1']) * self.x_scale), int((row['y2'] - row['y1']) * self.y_scale))
            if frame_id not in annotations:
                annotations[frame_id] = []
            annotations[frame_id].append((bbox, row['track_id'], row['filename'], row['confidence'], row['class_id']))  # Store track_id with bbox
        
        return annotations

    def _setup_tkinter_ui(self):
        self.root = tk.Tk()
        self.root.title("Interactive Bounding Boxes")

        self.canvas = Canvas(self.root, width=self.resized_width, height=self.resized_height)
        self.canvas.pack()

        self.control_window = tk.Toplevel(self.root)
        self.control_window.title("Frame Controls")

        self.next_button = tk.Button(self.control_window, text="Next Frame", command=self.next_frame)
        self.next_button.pack(side="left", padx=10)

        self.previous_button = tk.Button(self.control_window, text="Previous Frame", command=self.previous_frame)
        self.previous_button.pack(side="left", padx=10)

        self.jump_label = tk.Label(self.control_window, text="Jump to Frame:")
        self.jump_label.pack(side="left", padx=5)

        self.jump_entry = tk.Entry(self.control_window)
        self.jump_entry.pack(side="left", padx=5)
        self.jump_button = tk.Button(self.control_window, text="Go", command=self.jump_to_frame)
        self.jump_button.pack(side="left", padx=10)

        self.delete_label = tk.Label(self.control_window, text="Enter Track ID to Delete:")
        self.delete_label.pack(side="left", padx=5)

        self.delete_entry = tk.Entry(self.control_window)
        self.delete_entry.pack(side="left", padx=5)

        self.delete_button = tk.Button(self.control_window, text="Delete BBox", command=self.delete_bbox)
        self.delete_button.pack(side="left", padx=10)

        self.save_button = tk.Button(self.control_window, text="Save Changes", command=self.save_annotations)
        self.save_button.pack(side="left", padx=10)

        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.update_frame()

    def delete_bbox(self):
        track_id_to_delete = self.delete_entry.get()
        if not track_id_to_delete:
            print("Please enter a track ID.")
            return

        try:
            track_id_to_delete = int(track_id_to_delete)  # Convert input to an integer
        except ValueError:
            print("Invalid track ID.")
            return

        if self.current_frame in self.frame_annotations:
            # Search for the bounding box with the specified track_id
            self.bboxes = [bbox for bbox in self.bboxes if bbox[1] != track_id_to_delete]
            self.frame_annotations[self.current_frame] = [(bbox, track_id, filename, conf, class_id) for bbox, track_id, filename, conf, class_id in self.frame_annotations[self.current_frame] if track_id != track_id_to_delete]

            self.update_frame()  # Refresh the frame after deletion
            print(f"Deleted bounding box with Track ID {track_id_to_delete} from frame {self.current_frame}")
        else:
            print(f"No annotations found for frame {self.current_frame}.")

    def is_inside_bbox(self, x, y, bbox):
        bx, by, bw, bh = bbox
        return bx <= x <= bx + bw and by <= y <= by + bh

    def is_on_edge(self, x, y, bbox):
        bx, by, bw, bh = bbox
        return (bx + bw - 10 <= x <= bx + bw and by + bh - 10 <= y <= by + bh)

    def on_click(self, event):
        x, y = event.x, event.y
        for i, dets in enumerate(self.bboxes):
            bbox = dets[0]
            if self.is_on_edge(x, y, bbox):
                self.selected_box = i
                self.resizing = True
                break
            elif self.is_inside_bbox(x, y, bbox):
                self.selected_box = i
                self.start_x, self.start_y = x - bbox[0], y - bbox[1]
                self.moving = True
                break

    def on_drag(self, event):
        x, y = event.x, event.y
        if self.moving and self.selected_box is not None:
            bx, by, bw, bh = self.bboxes[self.selected_box][0]
            dets = self.bboxes[self.selected_box]
            bboxes = (x - self.start_x, y - self.start_y, bw, bh)
            self.bboxes[self.selected_box] = (bboxes, dets[1], dets[2], dets[3], dets[4])
            #self.bboxes[self.selected_box] = ((x - self.start_x, y - self.start_y, bw, bh), self.bboxes[self.selected_box][1])
            self.update_frame()
        elif self.resizing and self.selected_box is not None:
            bx, by, bw, bh = self.bboxes[self.selected_box][0]
            dets = self.bboxes[self.selected_box]
            bboxes = (bx, by, max(20, x - bx), max(20, y - by))
            self.bboxes[self.selected_box] = (bboxes, dets[1], dets[2], dets[3], dets[4])
            #self.bboxes[self.selected_box] = ((bx, by, max(20, x - bx), max(20, y - by)), self.bboxes[self.selected_box][1])
            self.update_frame()

    def on_release(self, event):
        self.moving = False
        self.resizing = False
        self.selected_box = None

    def update_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_resized = cv2.resize(frame, (self.resized_width, self.resized_height))

        for bbox, track_id, _, _, _ in self.bboxes:
            x, y, w, h = bbox
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame_resized, (x + w - 10, y + h - 10), (x + w, y + h), (0, 0, 255), -1)
            cv2.putText(frame_resized, f"ID: {int(track_id)}", (x + w + 5, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        frame_image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        frame_tk = ImageTk.PhotoImage(frame_image)

        self.canvas.create_image(0, 0, anchor="nw", image=frame_tk)
        self.canvas.image = frame_tk

    def next_frame(self):
        self.current_frame += 1
        if self.current_frame >= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        self.bboxes = self.frame_annotations.get(self.current_frame, [])
        self.update_frame()

    def previous_frame(self):
        self.current_frame -= 1
        if self.current_frame < 0:
            self.current_frame = 0

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        self.bboxes = self.frame_annotations.get(self.current_frame, [])
        self.update_frame()

    def jump_to_frame(self):
        try:
            frame_number = int(self.jump_entry.get())
            if 0 <= frame_number < int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                self.current_frame = frame_number
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    return

                self.bboxes = self.frame_annotations.get(self.current_frame, [])
                self.update_frame()
        except ValueError:
            pass  # Ignore if input is not a valid integer

    def save_annotations(self):
        updated_data = []

        # Update the annotations with the current bounding boxes
        for frame_id, dets in self.frame_annotations.items():
            for bbox, track_id, filename, conf, class_id in dets:
                x, y, w, h = bbox
                # Convert back to original resolution
                x1 = int(x / self.x_scale)
                y1 = int(y / self.y_scale)
                x2 = int((x + w) / self.x_scale)
                y2 = int((y + h) / self.y_scale)

                updated_data.append({
                    'filename': filename,
                    'frame_id': frame_id,
                    'track_id': track_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': conf,
                    'class_id': class_id
                })

        updated_df = pd.DataFrame(updated_data)
        updated_df.to_csv('updated_bboxes.csv', index=False)
        print("Annotations saved to updated_bboxes.csv")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoAnnotationApp("/home/tomass/tomass/Cam_record/04.09.25_2/perspective_views_fisheye_record_1756987992.9123657_frames_refined/right_video.avi", "/home/tomass/tomass/Cam_record/04.09.25_2/perspective_views_fisheye_record_1756987992.9123657_frames_refined/right/annotations.csv")
    app.run()
