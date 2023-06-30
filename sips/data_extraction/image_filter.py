from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from rosbags.typesys.types import sensor_msgs__msg__Image as ImageMessage


def cosine_similarity(frame_normalized_1, frame_normalized_2) -> float:
    return np.sum(frame_normalized_1 * frame_normalized_2)


class RedundantImageFilter:
    """
    For filtering images that look too similar.
    """

    def __init__(
        self,
        blur: str | None = "gaussian",
        blur_size: float = 5,
        blur_std: float = 2.5,
        threshold: float = 0.95,
    ) -> None:
        assert blur in [None, "gaussian", "median", "bilateral"]
        assert 0 <= threshold <= 1, "Threshold must be within [0, 1]"
        self.blur = blur
        self.blur_size = blur_size
        self.blur_std = blur_std
        self.threshold = threshold  # Higher values will save more data, where 1.0 saves everything and 0.0 saves nothing.

        self.n_saved = 0
        self.n_redundant = 0
        self.current_frame_norm: npt.NDArray[np.float_] | None = None

    def image_redundant(
        self, message: ImageMessage, save_path: str | Path, save_ims: bool
    ) -> bool:
        """
        Calculates if image is redundant and stores the image regardless of redundancy

        Parameters
        ----------
        message : ImageMessage
            input image
        save_path : str | Path
            path to location where image should be stored
        save_ims : bool
            indicates if image should be stored. depends on if the images of this
            bag were stored before

        Returns
        -------
        bool
            True if image is redundant
        """
        keyframe = message.data.reshape((message.height, message.width))
        # Save all images regardless of redundancy if they were not stored before
        if save_ims:
            Image.fromarray(keyframe, mode="L").save(save_path)

        # Skip if image is redundant
        if self._is_keyframe_redundant(keyframe):
            self.n_redundant += 1
            return True

        self.n_saved += 1
        return False

    def _is_keyframe_redundant(self, frame: npt.NDArray[np.uint8]) -> bool:
        # Subsample the frame to a lower resolution for the sake of filtering
        frame_resized: npt.NDArray[np.uint8] = cv2.resize(frame, (512, 512))  # type: ignore

        # Blur the frame to prevent high frequency sonar noise to influence the frame
        # content distance measurements.  For reference on different blurring methods
        # see https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
        frame_blurred: npt.NDArray[np.uint8]
        if self.blur is None:
            frame_blurred = frame_resized
        elif self.blur == "gaussian":
            frame_blurred = cv2.GaussianBlur(
                frame_resized, (self.blur_size, self.blur_size), self.blur_std
            )
        elif self.blur == "median":
            frame_blurred = cv2.medianBlur(frame_resized, self.blur_size)
        elif self.blur == "bilateral":
            frame_blurred = cv2.bilateralFilter(frame_resized, 9, 75, 75)
        else:
            raise ValueError(f"Invalid blur type '{self.blur}'")

        # L2 normalization of the frame as a vector, to prepare for cosine similarity
        frame_norm = frame_blurred / np.linalg.norm(frame_blurred)
        if (
            self.current_frame_norm is None
            or cosine_similarity(self.current_frame_norm, frame_norm) < self.threshold
        ):
            # Frame is not redundant
            self.current_frame_norm = frame_norm
            return False
        else:
            # Frame is redundant
            return True
