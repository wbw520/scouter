# Copyright (c) Karl Schulz, Leon Sixt
#
# All rights reserved.
#
# This code is licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import numpy as np
from skimage.transform import resize


# this module should be independent of torch and tensorflow
assert 'torch' not in globals()
assert 'tf' not in globals()
assert 'tensorflow' not in globals()


class WelfordEstimator:
    """
    Estimates the mean and standard derivation.
    For the algorithm see `wikipedia <https://en.wikipedia.org/wiki/
    Algorithms_for_calculating_variance#/Welford's_online_algorithm>`_.

    Example:
        Given a batch of images ``imgs`` with shape ``(10, 3, 64, 64)``, the mean and std could
        be estimated as follows::

            # exemplary data source: 5 batches of size 10, filled with random data
            batch_generator = (torch.randn(10, 3, 64, 64) for _ in range(5))

            estim = WelfordEstimator(3, 64, 64)
            for batch in batch_generator:
                estim(batch)

            # returns the estimated mean
            estim.mean()

            # returns the estimated std
            estim.std()

            # returns the number of seen samples, here 10
            estim.n_samples()

            # returns a mask with active neurons
            estim.active_neurons()
    """
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        """Resets the estimates."""
        self.m = None
        self.s = None
        self._n_samples = 0
        self._neuron_nonzero = None

    def fit(self, x):
        """ Update estimates without altering x """
        if self._n_samples == 0:
            # Initialize on first datapoint
            shape = x.shape[1:]
            self.m = np.zeros(shape)
            self.s = np.zeros(shape)
            self._neuron_nonzero = np.zeros(shape, dtype='long')
        for xi in x:
            self._neuron_nonzero += (xi != 0.)
            old_m = self.m.copy()
            self.m = self.m + (xi-self.m) / (self._n_samples + 1)
            self.s = self.s + (xi-self.m) * (xi-old_m)
            self._n_samples += 1
        return x

    def n_samples(self):
        """ Returns the number of seen samples. """
        return self._n_samples

    def mean(self):
        """ Returns the estimate of the mean. """
        return self.m

    def std(self):
        """ Returns the estimate of the standard derivation."""
        return np.sqrt(self.s / (self._n_samples - 1))

    def active_neurons(self, threshold=0.01):
        """
        Returns a mask of all active neurons.
        A neuron is considered active if ``n_nonzero / n_samples  > threshold``
        """
        return (self._neuron_nonzero.astype(np.float32) / self._n_samples) > threshold

    def state_dict(self):
        """ Returns internal state. Useful for saving to disk."""
        return {
            'm': self.m,
            's': self.s,
            'n_samples': self._n_samples,
            'neuron_nonzero': self._neuron_nonzero,
        }

    def load_state_dict(self, state):
        """ Loads the internal state of the estimator. """
        self.m = state['m']
        self.s = state['s']
        self._n_samples = state['n_samples']
        self._neuron_nonzero = state['neuron_nonzero']


def _to_saliency_map(capacity, shape=None, data_format='channels_last'):
    """
    Converts the layer capacity (in nats) to a saliency map (in bits) of the given shape.
    PyTorch:    Use data_format == 'channels_first'
    Tensorflow: Use data_format == 'channels_last'
    """
    if data_format == 'channels_first':
        saliency_map = np.nansum(capacity, 0)
    elif data_format == 'channels_last':
        saliency_map = np.nansum(capacity, -1)
    else:
        raise ValueError

    # to bits
    saliency_map /= float(np.log(2))

    if shape is not None:
        ho, wo = saliency_map.shape
        h, w = shape
        # Scale bits to the pixels
        saliency_map *= (ho*wo) / (h*w)
        return resize(saliency_map, shape, order=1, preserve_range=True)
    else:
        return saliency_map


def get_tqdm():
    """Tries to import ``tqdm`` from ``tqdm.auto`` if fails uses cli ``tqdm``."""
    try:
        from tqdm.auto import tqdm
        return tqdm
    except ImportError:
        from tqdm import tqdm
        return tqdm


def ifnone(a, b):
    """If a is None return b."""
    if a is None:
        return b
    else:
        return a


def to_unit_interval(x):
    """Scales ``x`` to be in ``[0, 1]``."""
    return (x - x.min()) / (x.max() - x.min())


def load_monkeys(center_crop=True, size=224, pil=False):
    """Returns the monkey test image."""
    from urllib.request import urlopen
    from io import BytesIO
    from PIL import Image

    if size is not None and type(size) == int:
        size = (size, size)

    resp = urlopen("http://farm1.static.flickr.com/95/247213534_e8be5222be.jpg")
    img_bytes = resp.read()
    img = Image.open(BytesIO(img_bytes))
    target = 382
    if pil:
        return img, target

    w, h = img.size
    cl = (w - h) // 2
    if center_crop:
        img = img.crop((cl, 0, cl + h, h))
    if size is not None:
        img = img.resize(size)
    return np.array(img), target


def plot_saliency_map(saliency_map, img=None, ax=None,
                      colorbar_label='Bits / Pixel',
                      colorbar_fontsize=14,
                      min_alpha=0.2, max_alpha=0.7, vmax=None,
                      colorbar_size=0.3, colorbar_pad=0.08):

    """
    Plots the heatmap with an bits/pixel colorbar and optionally overlays the image.

    Args:
        saliency_map (np.ndarray): the saliency_map.
        img (np.ndarray):  show this image under the saliency_map.
        ax: matplotlib axis. If ``None``, a new plot is created.
        colorbar_label (str): label for the colorbar.
        colorbar_fontsize (int): fontsize of the colorbar label.
        min_alpha (float): minimum alpha value for the overlay. only used if ``img`` is given.
        max_alpha (float): maximum alpha value for the overlay. only used if ``img`` is given.
        vmax: maximum value for colorbar.
        colorbar_size: width of the colorbar. default: Fixed(0.3).
        colorbar_pad: width of the colorbar. default: Fixed(0.08).

    Returns:
        The matplotlib axis ``ax``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from mpl_toolkits.axes_grid1.axes_size import Fixed
    from skimage.color import rgb2grey, grey2rgb
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

    if img is not None:
        # Underlay the image as greyscale
        grey = grey2rgb(rgb2grey(img))
        ax.imshow(grey)

    ax1_divider = make_axes_locatable(ax)
    if type(colorbar_size) == float:
        colorbar_size = Fixed(colorbar_size)
    if type(colorbar_pad) == float:
        colorbar_pad = Fixed(colorbar_pad)
    cax1 = ax1_divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
    if vmax is None:
        vmax = saliency_map.max()
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    n = 256
    half_jet_rgba = plt.cm.seismic(np.linspace(0.5, 1, n))
    half_jet_rgba[:, -1] = np.linspace(0.2, 1, n)
    cmap = mpl.colors.ListedColormap(half_jet_rgba)
    hmap_jet = cmap(norm(saliency_map))
    if img is not None:
        hmap_jet[:, :, -1] = (max_alpha - min_alpha)*norm(saliency_map) + min_alpha
    ax.imshow(hmap_jet, alpha=max_alpha)
    cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm)
    cbar.set_label(colorbar_label, fontsize=colorbar_fontsize)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid('off')
    ax.set_frame_on(False)
    plt.savefig('temp.jpg')
    return ax
