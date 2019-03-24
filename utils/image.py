from typing import Tuple

import cv2
import numpy as np


def factor_crop(image: np.ndarray, dest_size,
                factor: int = 32, padding: int = 0, based: str = 'min') \
        -> Tuple[np.ndarray, float, tuple]:

    def closest(num: int) \
            -> int:
        return int(np.ceil(float(num) / factor)) * factor

    base = {
        'min': np.min(image.shape[0:2]),
        'max': np.max(image.shape[0:2]),
        'w': image.shape[1],
        'h': image.shape[0],
    }

    scale = float(dest_size) / base.get(based, base['min'])

    # Scale the image
    image = cv2.resize(image, None, fx=scale, fy=scale)

    # Compute the padded image shape
    # Ensure it's divisible by factor
    h, w, *_ = image.shape
    nh, nw = closest(h), closest(w)
    new_shape = [nh, nw] if image.ndim < 3 else [nh, nw, image.shape[-1]]

    # Pad the image
    padded = np.full(new_shape, fill_value=padding, dtype=image.dtype)
    padded[0:h, 0:w] = image

    return padded, scale, image.shape
