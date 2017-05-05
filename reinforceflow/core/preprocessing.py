from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray


class NullPreprocessor(object):
    def __init__(self):
        pass

    def __call__(self, frame, reset=False):
        return frame


class RGBFramePreprocessor(object):
    def __init__(self, stack_len, height, width, to_gray=True):
        self.stack_len = stack_len
        self.height = height
        self.width = width
        self.to_gray = to_gray
        self.shape = (1, height, width, stack_len if to_gray else stack_len*3)
        self._frame_stack = None
        self._prev_frame = None

    def __call__(self, frame, reset=False):
        frame = self._preprocess(frame)
        if not self.stack_len or self.stack_len == 1:
            return frame
        return self._stack_frames(frame, reset)

    def _preprocess(self, frame):
        if self.to_gray:
            frame = rgb2gray(frame)
        if self.height and self.width:
            frame = resize(frame, (self.height, self.width))
        frame = frame.reshape(1, *frame.shape, 1)
        return frame

    def _stack_frames(self, frame, reset):
        if reset or self._frame_stack is None:
            self._frame_stack = np.repeat(frame, self.stack_len, axis=3)
        else:
            self._frame_stack = np.append(frame, self._frame_stack[:, :, :, :self.stack_len - 1], axis=3)
        return self._frame_stack


class AtariPreprocessor(RGBFramePreprocessor):
    def __init__(self, stack_len, height=84, width=84, to_gray=True, max_merge_frames=True):
        super(AtariPreprocessor, self).__init__(stack_len, height, width, to_gray)
        self.max_merge_frames = max_merge_frames

    def __call__(self, frame, reset=False):
        frame = self._preprocess(frame)
        # Takes maximum value for each pixel value over the current and previous frame.
        # Used to get around Atari sprites flickering (see Mnih et al. (2015)).
        if self.max_merge_frames and not reset:
            prev_frame = self._prev_frame
            self._prev_frame = frame
            frame = np.maximum.reduce([frame, prev_frame]) if prev_frame else frame

        if not self.stack_len or self.stack_len == 1:
            return frame
        return self._stack_frames(frame, reset)
