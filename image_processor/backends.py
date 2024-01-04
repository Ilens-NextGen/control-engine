import cv2
import numpy as np
from io import RawIOBase
from typing import List, Optional, Tuple
from datetime import datetime
import os
from django.core.files.uploadedfile import UploadedFile
from math import sqrt, ceil, floor

def create_image_strip(frames: list, file=False) -> Optional[str]:
    """Create a long single image strip from a group of similar images
    Args:
        frames (List): A list of images

    Returns:
        str: The file path to the image strip
    """
    # Concatenate the frames horizontally
    strip = np.concatenate(frames, axis=1)

    # Display the strip
    output_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_name = f"strip-{output_timestamp}.jpg"
    cv2.imwrite(output_name, strip)
    return output_name

def split_video_to_image_strip(video: str) -> str:
    """Converts a video into an image strip
    Args:
        video (RawIOBase): A video
    Returns: 
        str: the file path to the image strip
    """
    cap = cv2.VideoCapture(video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(3): #range(num_frames):
        ret, frame = cap.read()

        if ret:
            print(f"Adding frame {i}")
            frame = cv2.resize(frame, None, fx=0.6, fy=0.6)

            frames.append(frame)
    strip = np.concatenate(frames, axis=1)

    # Display the strip

    cv2.imwrite(output_name, strip)
    return output_name

class ImageProcessor:
    """A class for splitting and manipulating the images and videos we'll get from the
    frontend, saving it, etc"""
    
    def convert_videos_to_frames(self, video: str | UploadedFile, max_frames = 4) -> Tuple[list, int, int]:
        if isinstance(video, str):
            cap = cv2.VideoCapture(video)
        else:
            video_file_path = video_obj.temporary_file_path()
            cap = cv2.VideoCapture(video_file_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            fps = int(num_frames / max_frames)
        else:
            fps = int(cap.get(cv2.CAP_PROP_FPS)) 
        time_in_s = max_frames or int(num_frames / fps)
        f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ff_width = int(f_width * 0.8)
        ff_height = int(f_height * 0.8)
        frames = []
        count = 1
        for i in range(num_frames):
            if count > time_in_s:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if i == (count * fps):
                print(f"Adding frame {i}")
                frame = cv2.resize(frame, (ff_width, ff_height))
                frames.append(frame)
                count += 1
        return frames, ff_width, ff_height
   
    def convert_image_list_to_frames(self, images: List[str] | List[UploadedFile]= []):
        """Converts a list of images to frames"""
        frames = []
        if all(isinstance(image, str) for image in images):
            for image in images:
                frame = cv2.imread(image)
                frames.append(frame)
        else: # Images are straight from fileStorage
            for file in images:
                data = file.read()
                # file.seek(0) Incase we would like to use this for anything else
                frame = np.asarray(bytearray(data), dtype="uint8")
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                frames.append(frame)
        return frames

    def calculate_grid_positions(self, num_frames: int, frame_width: int, frame_height: int) -> Tuple[int, int]:
        """Calculate the grid positions for overlaying the frames
        Args:
            num_frames (int): the number of frames
            frame_width (int): the width of a frame
            frame_height (int): the height of a frame
        Returns:
            Tuple[int, int]: the rows - column value
        """
        root = sqrt(num_frames)
        if frame_width > frame_height:  # Landscape orientation
            rows = 2
            cols = num_frames / 2
        else:  # Portrait orientation
            rows = num_frames / 2
            cols = 2

        # Adjust if necessary
        while rows * cols < num_frames:
            if frame_width > frame_height:  # Landscape orientation
                cols += 1
            else:  # Portrait orientation
                rows += 1

        return int(rows), int(cols)
    def convert_frames_to_grid(self, frames: list, rows: int, cols: int):
        """Converts the frames into a grid of images"""
        # Reshape the list of frames into a grid
        grid = []
        temp = []
        for i, frame in enumerate(frames):
            if i % cols == 0:
                if len(temp) > 0:
                    grid.append(temp.copy())
                temp = []
            temp.append(frame)
        if len(temp) > 0:
	        grid.append(temp)
        # grid = np.reshape(frames, (rows, cols))

        # Concatenate the frames along the vertical and horizontal axes to create the final image
        final_image = np.concatenate([np.concatenate(row_frames, axis=1) for row_frames in grid], axis=0)

        return final_image

    def save_image_grid(self, image, output_path: str, show = True, save=False):
        output_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_name = f"output/grid-{output_timestamp}.jpg"
        output_path = os.path.abspath(output_name)
        print(f"Saving image grid to {output_path}")
        if show:
            cv2.imshow('Image Grid', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save:
            cv2.imwrite(output_name, image)
        return output_name
        
    def video_to_grid(self, video, show, max_frames=4):
        frames, width, height = self.convert_videos_to_frames(video, max_frames)
        rows, cols = self.calculate_grid_positions(len(frames), width, height)
        image_grid = self.convert_frames_to_grid(frames, rows, cols)
        return self.save_image_grid(image_grid, show)
        
        
ImageProcessor().video_to_grid("test1.mp4", True)
