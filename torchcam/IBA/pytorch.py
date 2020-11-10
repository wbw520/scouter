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
import torch.nn as nn
import torch
import warnings
from contextlib import contextmanager
from torchvision.transforms import Normalize, Compose
from torchcam.IBA.utils import _to_saliency_map, get_tqdm, ifnone

# Helper Functions


def to_saliency_map(capacity, shape=None):
    """
    Converts the layer capacity (in nats) to a saliency map (in bits) of the given shape .

    Args:
        capacity (np.ndarray): Capacity in nats.
        shape (tuple): (height, width) of the image.
    """
    return _to_saliency_map(capacity, shape, data_format="channels_first")


def insert_into_sequential(sequential, layer, idx):
    """
    Returns a ``nn.Sequential`` with ``layer`` inserted in ``sequential`` at position ``idx``.
    """
    children = list(sequential.children())
    children.insert(idx, layer)
    return nn.Sequential(*children)


def tensor_to_np_img(img_t):
    """
    Convert a torch tensor of shape ``(c, h, w)`` to a numpy array of shape ``(h, w, c)``
    and reverse the torchvision prepocessing.
    """
    return Compose([
        Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        Normalize(std=[1, 1, 1], mean=[-0.485, -0.456, -0.406]),
    ])(img_t).detach().cpu().numpy().transpose(1, 2, 0)


def imagenet_transform(resize=256, crop_size=224):
    """Returns the default torchvision imagenet transform. """
    from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize
    return Compose([
        Resize(resize),
        CenterCrop(crop_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_imagenet_folder(path, image_size=260, transform='default'):
    """
    Returns a ``torchvision.datasets.ImageFolder`` with the default
    torchvision preprocessing.
    """
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize
    if transform == 'default':
        transform = Compose([
            CenterCrop(256), Resize(image_size), ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return ImageFolder(path, transform=transform)


class _SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with fixed gaussian kernels, used to smoothen the input """
    def __init__(self, kernel_size, sigma, channels,):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, \
            "kernel_size must be an odd number (for padding), {} given".format(self.kernel_size)
        variance = sigma ** 2.
        x_cord = torch.arange(kernel_size, dtype=torch.float)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * np.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean_xy) ** 2., dim=-1) /
            (2 * variance)
        )
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_3d = kernel_2d.expand(channels, 1, -1, -1)  # expand in channel dimension
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels,
                              padding=0, kernel_size=kernel_size,
                              groups=channels, bias=False)
        self.conv.weight.data.copy_(kernel_3d)
        self.conv.weight.requires_grad = False
        self.pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))

    def parameters(self, **kwargs):
        """returns no parameters"""
        return []

    def forward(self, x):
        return self.conv(self.pad(x))


class TorchWelfordEstimator(nn.Module):
    """
    Estimates the mean and standard derivation.
    For the algorithm see ``https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance``.

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

            # returns the number of samples, here 10
            estim.n_samples()

            # returns a mask with active neurons
            estim.active_neurons()
    """
    def __init__(self):
        super().__init__()
        self.device = None  # Defined on first forward pass
        self.shape = None  # Defined on first forward pass
        self.register_buffer('_n_samples', torch.tensor([0], dtype=torch.long))

    def _init(self, shape, device):
        self.device = device
        self.shape = shape
        self.register_buffer('m', torch.zeros(*shape))
        self.register_buffer('s', torch.zeros(*shape))
        self.register_buffer('_neuron_nonzero', torch.zeros(*shape, dtype=torch.long))
        self.to(device)

    def forward(self, x):
        """ Update estimates without altering x """
        if self.shape is None:
            # Initialize runnnig mean and std on first datapoint
            self._init(x.shape[1:], x.device)
        for xi in x:
            self._neuron_nonzero += (xi != 0.).long()
            old_m = self.m.clone()
            self.m = self.m + (xi-self.m) / (self._n_samples.float() + 1)
            self.s = self.s + (xi-self.m) * (xi-old_m)
            self._n_samples += 1
        return x

    def n_samples(self):
        """ Returns the number of seen samples. """
        return int(self._n_samples.item())

    def mean(self):
        """ Returns the estimate of the mean. """
        return self.m

    def std(self):
        """returns the estimate of the standard derivation."""
        return torch.sqrt(self.s / (self._n_samples.float() - 1))

    def active_neurons(self, threshold=0.01):
        """
        Returns a mask of all active neurons.
        A neuron is considered active if ``n_nonzero / n_samples  > threshold``
        """
        return (self._neuron_nonzero.float() / self._n_samples.float()) > threshold


class _InterruptExecution(Exception):
    pass


class _IBAForwardHook:
    def __init__(self, iba, input_or_output="output"):
        self.iba = iba
        self.input_or_output = input_or_output

    def __call__(self, m, inputs, outputs):
        if self.input_or_output == "input":
            return self.iba(inputs)
        elif self.input_or_output == "output":
            return self.iba(outputs)


class IBA(nn.Module):
    """
    IBA finds relevant features of your model by applying noise to
    intermediate features.

    Example: ::

        model = Net()
        # Create the Per-Sample Bottleneck:
        iba = IBA(model.conv4)

        # Estimate the mean and variance.
        iba.estimate(model, datagen)

        img, target = next(iter(datagen(batch_size=1)))

        # Closure that returns the loss for one batch
        model_loss_closure = lambda x: F.nll_loss(F.log_softmax(model(x), target)

        # Explain class target for the given image
        saliency_map = iba.analyze(img.to(dev), model_loss_closure)
        plot_saliency_map(img.to(dev))


    Args:
        layer: The layer after which to inject the bottleneck
        sigma: The standard deviation of the gaussian kernel to smooth
            the mask, or None for no smoothing
        beta: Weighting of model loss and mean information loss.
        min_std: Minimum std of the features
        lr: Optimizer learning rate. default: 1. As we are optimizing
            over very few iterations, a relatively high learning rate
            can be used compared to the training of the model itself.
        batch_size: Number of samples to use per iteration
        input_or_output: Select either ``"output"`` or ``"input"``.
        initial_alpha: Initial value for the parameter.
    """
    def __init__(self,
                 layer=None,
                 sigma=1.,
                 beta=10,
                 min_std=0.01,
                 optimization_steps=10,
                 lr=1,
                 batch_size=10,
                 initial_alpha=5.0,
                 active_neurons_threshold=0.01,
                 feature_mean=None,
                 feature_std=None,
                 estimator=None,
                 progbar=False,
                 input_or_output="output",
                 relu=False):
        super().__init__()
        self.relu = relu
        self.beta = beta
        self.min_std = min_std
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.batch_size = batch_size
        self.initial_alpha = initial_alpha
        self.alpha = None  # Initialized on first forward pass
        self.progbar = progbar
        self.sigmoid = nn.Sigmoid()
        self._buffer_capacity = None  # Filled on forward pass, used for loss
        self.sigma = sigma
        self.estimator = ifnone(estimator, TorchWelfordEstimator())
        self.device = None
        self._estimate = False
        self._mean = feature_mean
        self._std = feature_std
        self._active_neurons = None
        self._active_neurons_threshold = active_neurons_threshold
        self._restrict_flow = False
        self._interrupt_execution = False
        self._hook_handle = None

        # Check if modifying forward hooks are supported by the current torch version
        if layer is not None:
            try:
                from packaging import version
                if version.parse(torch.__version__) < version.parse("1.2"):
                    raise RuntimeWarning(
                        "IBA has to be manually injected into the model with your "
                        "version of torch: Forward hooks are only allowed to modify "
                        "the output in torch >= 1.2. Please upgrade torch or resort to "
                        "adding the IBA layer into the model directly as: model.any_layer = "
                        "nn.Sequential(model.any_layer, iba)")
            finally:
                pass  # Do not complain if packaging is not installed

            # self._hook_handle = layer.register_forward_hook(lambda m, x, y: self(y))

            # for handle, hooks in layer._forward_hooks.items():
            #     if type(hooks) == _IBAForwardHook:
            #         raise ValueError("Another IBA object is already attacted to the layer. "
            #                          "Remove it by calling `detach()`")

            # Attach the bottleneck after the model layer as forward hook
            self._hook_handle = layer.register_forward_hook(
                _IBAForwardHook(self, input_or_output))

        else:
            pass

    def _reset_alpha(self):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.alpha.fill_(self.initial_alpha)

    def _build(self):
        """
        Initialize alpha with the same shape as the features.
        We use the estimator to obtain shape and device.
        """
        if self.estimator.n_samples() <= 0:
            raise RuntimeWarning("You need to estimate the feature distribution"
                                 " before using the bottleneck.")
        shape = self.estimator.shape
        device = self.estimator.device
        self.alpha = nn.Parameter(torch.full(shape, self.initial_alpha, device=device),
                                  requires_grad=True)
        if self.sigma is not None and self.sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(2 * self.sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            self.smooth = _SpatialGaussianKernel(kernel_size, self.sigma, shape[0]).to(device)
        else:
            self.smooth = None

    def detach(self):
        """ Remove the bottleneck to restore the original model """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        else:
            raise ValueError("Cannot detach hock. Either you never attached or already detached.")

    def forward(self, x):
        """
        You don't need to call this method manually.

        The IBA acts as a model layer, passing the information in `x` along to the next layer
        either as-is or by restricting the flow of infomration.
        We use it also to estimate the distribution of `x` passing through the layer.
        """
        if self._restrict_flow:
            return self._do_restrict_information(x, self.alpha)
        if self._estimate:
            self.estimator(x)
        if self._interrupt_execution:
            raise _InterruptExecution()
        return x

    @contextmanager
    def interrupt_execution(self):
        """
        Interrupts the execution of the model, once PerSampleBottleneck is called. Useful
        for estimation when the model has only be executed until the Per-Sample Bottleneck.

        Example:
            Executes the model only until the bottleneck layer::

                with bltn.interrupt_execution():
                    out = model(x)
                    # out will not be defined
                    print("this will not be printed")
        """
        self._interrupt_execution = True
        try:
            yield
        except _InterruptExecution:
            pass
        finally:
            self._interrupt_execution = False

    @staticmethod
    def _calc_capacity(mu, log_var):
        """ Return the feature-wise KL-divergence of p(z|x) and q(z) """
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())

    @staticmethod
    def _kl_div(r, lambda_, mean_r, std_r):
        r_norm = (r - mean_r) / std_r
        var_z = (1 - lambda_) ** 2

        log_var_z = torch.log(var_z)

        mu_z = r_norm * lambda_

        capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
        return capacity

    def _do_restrict_information(self, x, alpha):
        """ Selectively remove information from x by applying noise """
        if alpha is None:
            raise RuntimeWarning("Alpha not initialized. Run _init() before using the bottleneck.")

        if self._mean is None:
            self._mean = self.estimator.mean()

        if self._std is None:
            self._std = self.estimator.std()

        if self._active_neurons is None:
            self._active_neurons = self.estimator.active_neurons()

        # Smoothen and expand alpha on batch dimension
        lamb = self.sigmoid(alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        self._buffer_capacity = self._kl_div(x, lamb, self._mean, self._std) * self._active_neurons

        eps = x.data.new(x.size()).normal_()
        ε = self._std * eps + self._mean
        λ = lamb
        z = λ * x + (1 - λ) * ε
        z *= self._active_neurons

        # Sample new output values from p(z|x)

        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)

        return z

    @contextmanager
    def enable_estimation(self):
        """
        Context manager to enable estimation of the mean and standard derivation.
        We recommend to use the `self.estimate` method.
        """
        self._estimate = True
        try:
            yield
        finally:
            self._estimate = False

    def reset_estimate(self):
        """
        Resets the estimator. Useful if the distribution changes. Which can happen if you
        trained the model more.
        """
        self.estimator = TorchWelfordEstimator()

    def estimate(self, model, dataloader, device=None, n_samples=10000, progbar=None, reset=True):
        """ Estimate mean and variance using the welford estimator.
            Usually, using 10.000 i.i.d. samples gives decent estimates.

            Args:
                model: the model containing the bottleneck layer
                dataloader: yielding ``batch``'s where the first sample
                    ``batch[0]`` is the image batch.
                device: images will be transfered to the device. If ``None``, it uses the device
                    of the first model parameter.
                n_samples (int): run the estimate on that many samples
                progbar (bool): show a progress bar.
                reset (bool): reset the current estimate of the mean and std

        """
        progbar = progbar if progbar is not None else self.progbar
        if progbar:
            try:
                tqdm = get_tqdm()
                bar = tqdm(dataloader, total=n_samples)
            except ImportError:
                warnings.warn("Cannot load tqdm! Sorry, no progress bar")
                bar = None
        else:
            bar = None

        if device is None:
            device = next(iter(model.parameters())).device
        if reset:
            self.reset_estimate()
        for batch in dataloader:
            imgs = batch[0]
            if self.estimator.n_samples() > n_samples:
                break
            with torch.no_grad(), self.interrupt_execution(), self.enable_estimation():
                model(imgs.to(device))
            if bar:
                bar.update(len(imgs))
        if bar:
            bar.close()

        # Cache results
        self._mean = self.estimator.mean()
        self._std = self.estimator.std()
        self._active_neurons = self.estimator.active_neurons(self._active_neurons_threshold).float()
        # After estimaton, feature map dimensions are known and
        # we can initialize alpha and the smoothing kernel
        if self.alpha is None:
            self._build()

    @contextmanager
    def restrict_flow(self):
        """
        Context mananger to enable information supression.

        Example:
            To make a prediction, with the information flow being supressed.::

                with iba.restrict_flow():
                    # now noise is added
                    model(x)
        """
        self._restrict_flow = True
        try:
            yield
        finally:
            self._restrict_flow = False

    def analyze(self, input_t, model_loss_fn, mode="saliency",
                beta=None, optimization_steps=None, min_std=None,
                lr=None, batch_size=None, active_neurons_threshold=0.01):
        """
        Generates a heatmap for a given sample. Make sure you estimated mean and variance of the
        input distribution.

        Args:
            input_t: input image of shape (1, C, H W)
            model_loss_fn: closure evaluating the model
            mode: how to post-process the resulting map: 'saliency' (default) or 'capacity'
            beta: if not None, overrides the bottleneck beta value
            optimization_steps: if not None, overrides the bottleneck optimization_steps value
            min_std: if not None, overrides the bottleneck min_std value
            lr: if not None, overrides the bottleneck lr value
            batch_size: if not None, overrides the bottleneck batch_size value
            active_neurons_threshold: used threshold to determine if a neuron is active

        Returns:
            The heatmap of the same shape as the ``input_t``.
        """
        assert input_t.shape[0] == 1, "We can only fit one sample a time"

        # TODO: is None
        beta = ifnone(beta, self.beta)
        optimization_steps = ifnone(optimization_steps, self.optimization_steps)
        min_std = ifnone(min_std, self.min_std)
        lr = ifnone(lr, self.lr)
        batch_size = ifnone(batch_size, self.batch_size)
        active_neurons_threshold = ifnone(active_neurons_threshold, self._active_neurons_threshold)

        batch = input_t.expand(batch_size, -1, -1, -1)

        # Reset from previous run or modifications
        self._reset_alpha()
        optimizer = torch.optim.Adam(lr=lr, params=[self.alpha])

        if self.estimator.n_samples() < 1000:
            warnings.warn(f"Selected estimator was only fitted on {self.estimator.n_samples()} "
                          f"samples. Might not be enough! We recommend 10.000 samples.")
        std = self.estimator.std()
        self._active_neurons = self.estimator.active_neurons(active_neurons_threshold).float()
        self._std = torch.max(std, min_std*torch.ones_like(std))

        self._loss = []
        self._alpha_grads = []
        self._model_loss = []
        self._information_loss = []

        opt_range = range(optimization_steps)
        try:
            tqdm = get_tqdm()
            opt_range = tqdm(opt_range, desc="Training Bottleneck", disable=not self.progbar)
        except ImportError:
            if self.progbar:
                warnings.warn("Cannot load tqdm! Sorry, no progress bar")
                self.progbar = False

        with self.restrict_flow():
            for _ in opt_range:
                optimizer.zero_grad()
                model_loss = model_loss_fn(batch)
                # Taking the mean is equivalent of scaling the sum with 1/K
                information_loss = self.capacity().mean()
                loss = model_loss + beta * information_loss
                loss.backward()
                optimizer.step()

                self._alpha_grads.append(self.alpha.grad.cpu().numpy())
                self._loss.append(loss.item())
                self._model_loss.append(model_loss.item())
                self._information_loss.append(information_loss.item())

        return self._get_saliency(mode=mode, shape=input_t.shape[2:])

    def capacity(self):
        """
        Returns a tensor with the capacity from the last input, averaged
        over the redundant batch dimension.
        Shape is ``(self.channels, self.height, self.width)``
        """
        return self._buffer_capacity.mean(dim=0)

    def _get_saliency(self, mode='saliency', shape=None):
        capacity_np = self.capacity().detach().cpu().numpy()
        if mode == "saliency":
            # In bits, summed over channels, scaled to input
            return to_saliency_map(capacity_np, shape)
        elif mode == "capacity":
            # In bits, not summed, not scaled
            return capacity_np / float(np.log(2))
        else:
            raise ValueError
