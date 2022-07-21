import cv2
import numpy as np

#         Based on PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/PySceneDetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# Copyright (C) 2014-2020 Brandon Castellano <http://www.bcastell.com>.
#



class SceneDetector:
    """Detects fast cuts using changes in colour and intensity between frames.
    """

    def __init__(self, threshold=30.0):
        # type: (float, Union[int, FrameTimecode]) -> None
        self.threshold = threshold
        # Minimum length of any given scene, in frames (int) or FrameTimecode

        self.last_frame = None
        self.last_scene_cut = None
        self.last_hsv = None
        self._metric_keys = ['content_val', 'delta_hue', 'delta_sat', 'delta_lum']
        self.cli_name = 'detect-content'
        
    def is_new_scene(self, frame_img):
        # type: (numpy.ndarray) -> Boolean
        """ Similar to ThresholdDetector, but using the HSV colour space DIFFERENCE instead
        of single-frame RGB/grayscale intensity (thus cannot detect slow fades with this method).
        Arguments:
            frame_img (numpy.ndarray): Decoded frame image (numpy.ndarray) to perform scene detection on.
        Returns:
            Boolean: True if frame belongs to a new scene, False otherwise.
        """

        cut_list = []
        metric_keys = self._metric_keys
        _unused = ''

        

        # We can only start detecting once we have a frame to compare with.
        if self.last_frame is not None:
            # Change in average of HSV (hsv), (h)ue only, (s)aturation only, (l)uminance only.
            # These are refered to in a statsfile as their respective self._metric_keys string.
            delta_hsv_avg, delta_h, delta_s, delta_v = 0.0, 0.0, 0.0, 0.0

          
            num_pixels = frame_img.shape[0] * frame_img.shape[1]
            curr_hsv = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
            last_hsv = self.last_hsv
            if not last_hsv:
                last_hsv = cv2.split(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2HSV))

            delta_hsv = [0, 0, 0, 0]
            for i in range(3):
                num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
                curr_hsv[i] = curr_hsv[i].astype(np.int32)
                last_hsv[i] = last_hsv[i].astype(np.int32)
                delta_hsv[i] = np.sum(
                    np.abs(curr_hsv[i] - last_hsv[i])) / float(num_pixels)
            delta_hsv[3] = sum(delta_hsv[0:3]) / 3.0
            delta_h, delta_s, delta_v, delta_hsv_avg = delta_hsv


            self.last_hsv = curr_hsv

            # We consider any frame over the threshold a new scene, but only if
            # the minimum scene length has been reached (otherwise it is ignored).
            if delta_hsv_avg >= self.threshold:
                return True

            if self.last_frame is not None and self.last_frame is not _unused:
                del self.last_frame

        
        self.last_frame = frame_img.copy()

        return False