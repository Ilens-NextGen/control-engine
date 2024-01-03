import cv2
import numpy as np
from io import RawIOBase
from typing import List, Optional
from datetime import datetime
import os


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
    main = []

    for i in range(3): #range(num_frames):
        ret, frame = cap.read()

        if ret:
            print(f"Adding frame {i}")
            frame = cv2.resize(frame, None, fx=0.6, fy=0.6)

            frames.append(frame)
    strip = np.concatenate(frames, axis=1)
    cv2.imshow('Image Strip', strip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Display the strip
    output_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_name = f"strip-{output_timestamp}.jpg"
    output_path = os.path.abspath(output_name)
    print(f"Saving image strip to {output_path}")
    cv2.imwrite(output_name, strip)
    return output_name
split_video_to_image_strip('test1.mp4')
