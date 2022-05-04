import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def dense_crf(probs, img=None, n_classes=2, n_iters=1, scale_factor=1):
    c, h, w = probs.shape
    if img is not None:
        assert img.shape[1:3] == (h, w)
        img = np.transpose(img, (1, 2, 0)).copy(order="C")

    d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80 / scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(n_iters)

    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w))
    return preds

