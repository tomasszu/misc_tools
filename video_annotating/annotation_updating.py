import tkinter as tk
from tkinter import filedialog
import pandas as pd
import cv2

# Define class to handle GUI and annotations
class AnnotationUpdater:
    def __init__(self, csv_path, video_path):
        # Load existing annotations from CSV
        self.df = pd.read_csv(csv_path)

        # Open the video
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            exit()

        # Initialize tkinter window
        self.window = tk.Tk()
        self.window.title("Annotation Updater")

        # Initialize current frame and annotations
        self.frame_id = 0
        self.bboxes = []  # To hold bounding boxes for the current frame
        self.track_ids = {}  # Store corrected track_ids
        self.entries = []  # Store references to the Tkinter Entry widgets

        # Load the first frame
        self.load_frame()

        # Create the UI buttons for navigating through frames
        self.create_buttons()

        cv2.waitKey(0)

    def load_frame(self):
        """ Load and display the current frame """
        # Set the video to the correct frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
        ret, frame = self.cap.read()

        if not ret:
            print(f"Error loading frame {self.frame_id} or end of video")
            return

        # print(f"Frame dimensions: {frame.shape}")  # Debug frame dimensions

        # Clear previous bounding boxes from the frame
        self.bboxes = self.get_bboxes_for_frame(self.frame_id)
        print(f"frame {self.frame_id}")  # Debug

        # Draw the bounding boxes
        self.display_frame_with_bboxes(frame)

        # Remove any existing entry widgets
        for entry in self.entries:
            entry.destroy()
        self.entries = []

        # Create entry widgets for each bounding box
        self.create_entry_widgets()

    def get_bboxes_for_frame(self, frame_id):
        """ Fetch bounding boxes for the given frame_id from the DataFrame """
        frame_data = self.df[self.df['frame_id'] == frame_id]
        bboxes = []
        for _, row in frame_data.iterrows():
            bbox = (int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2']), int(row['track_id']))
            bboxes.append(bbox)
        return bboxes

    def display_frame_with_bboxes(self, frame):
        """ Draw bounding boxes on the current frame """
        for bbox in self.bboxes:
            x1, y1, x2, y2, track_id = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 8)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 12)
        
        # Ensure frame is in correct format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_resized = cv2.resize(frame, (1440, 900))
        
        # Display the frame with bounding boxes
        cv2.imshow('Frame', frame_resized)

    def create_buttons(self):
        """ Create the frame navigation buttons """
        prev_button = tk.Button(self.window, text="Previous Frame", command=self.prev_frame)
        prev_button.pack(side=tk.LEFT)

        next_button = tk.Button(self.window, text="Next Frame", command=self.next_frame)
        next_button.pack(side=tk.LEFT)

        save_button = tk.Button(self.window, text="Save Annotations", command=self.save_annotations)
        save_button.pack(side=tk.LEFT)

        # Frame jump entry and button
        self.frame_entry = tk.Entry(self.window, width=10)
        self.frame_entry.pack(side=tk.LEFT)

        go_button = tk.Button(self.window, text="Go to Frame", command=self.go_to_frame)
        go_button.pack(side=tk.LEFT)

        # Track ID removal entry and button
        self.remove_id_entry = tk.Entry(self.window, width=10)
        self.remove_id_entry.pack(side=tk.LEFT)

        remove_button = tk.Button(self.window, text="Remove ID", command=self.remove_track_id)
        remove_button.pack(side=tk.LEFT)

    def create_entry_widgets(self):
        """ Create entry widgets for track ID editing """
        for i, bbox in enumerate(self.bboxes):
            x1, y1, x2, y2, track_id = bbox
            # Create an Entry widget for track ID
            entry = tk.Entry(self.window)
            entry.insert(tk.END, str(track_id))  # Set the current track ID as the default value
            entry.pack()
            self.entries.append(entry)

    def prev_frame(self):
        """ Show the previous frame and update annotations """
        if self.frame_id > 0:
            self.frame_id -= 1
            self.load_frame()
        
        cv2.waitKey(1)

    def next_frame(self):
        """ Show the next frame and update annotations """
        self.frame_id += 1
        self.load_frame()

        cv2.waitKey(1)

    def go_to_frame(self):
        """ Jump to a specific frame based on user input """
        try:
            frame_number = int(self.frame_entry.get())
            if 0 <= frame_number < int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):  # Validate frame range
                self.frame_id = frame_number
                self.load_frame()
            else:
                print("Invalid frame number. Out of range.")
        except ValueError:
            print("Invalid input. Please enter a number.")

        cv2.waitKey(1)

    def remove_track_id(self):
        """ Remove all occurrences of a specified track_id from the current frame onward """
        try:
            track_id_to_remove = int(self.remove_id_entry.get())
            
            # Filter out rows where track_id matches from the current frame onwards
            initial_size = len(self.df)
            self.df = self.df[~((self.df['frame_id'] >= self.frame_id) & (self.df['track_id'] == track_id_to_remove))]
            
            updated_size = len(self.df)
            print(f"Removed track_id {track_id_to_remove} from frame {self.frame_id} onwards. {initial_size - updated_size} entries removed.")

            # Reload frame to reflect the changes
            self.load_frame()
        except ValueError:
            print("Invalid input. Please enter a valid track ID.")

    def save_annotations(self):
        """ Save the updated annotations back to the CSV """
        updated_track_ids = {}  # Track changes made

        for i, bbox in enumerate(self.bboxes):
            x1, y1, x2, y2, track_id = bbox
            new_track_id = self.entries[i].get()  # Get the new track ID from the entry widget
            
            if new_track_id.isdigit():
                new_track_id = int(new_track_id)  # Convert to integer
            else:
                continue  # Ignore invalid entries

            print(f"Old: {track_id}, New: {new_track_id}")  # Debugging output

            if new_track_id != track_id:
                updated_track_ids[track_id] = new_track_id  # Store for updating bboxes
                self.df.loc[(self.df['frame_id'] >= self.frame_id) & (self.df['track_id'] == track_id), 'track_id'] = new_track_id

        # Apply updates to self.bboxes to match the DataFrame updates
        for i in range(len(self.bboxes)):
            x1, y1, x2, y2, track_id = self.bboxes[i]
            if track_id in updated_track_ids:
                self.bboxes[i] = (x1, y1, x2, y2, updated_track_ids[track_id])  # Update displayed bounding boxes
        
        # Save updated CSV
        self.df.to_csv("updated_annotations.csv", index=False)
        print("Annotations saved to updated_annotations.csv")

# Create an instance of the AnnotationUpdater
csv_path = '/home/tomass/tomass/Cam_record/04.09.25_2/perspective_views_fisheye_record_1756987992.9123657_frames_refined/right/annotations.csv'
video_path = '/home/tomass/tomass/Cam_record/04.09.25_2/perspective_views_fisheye_record_1756987992.9123657_frames_refined/right_video.avi'
annotation_updater = AnnotationUpdater(csv_path, video_path)

# Start the tkinter GUI loop
annotation_updater.window.mainloop()
