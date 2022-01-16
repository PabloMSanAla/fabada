"""
FABADA is a non-parametric noise reduction technique based on Bayesian
inference that iteratively evaluates possible smoothed  models  of
the  data introduced,  obtaining  an  estimation  of the  underlying
signal that is statistically  compatible  with the  noisy  measurements.

based on P.M. Sanchez-Alarcon, Y. Ascasibar, 2022
"Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"

Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.


Instructions:
Save the code as a .pyw file to disable the console. Save as a .py to enable it.
Install the latest miniforge for you into a folder, don't add it to path, launch it's command line from start menu.
Note: if python is installed elsewhere this may fail. If it fails, try this again with miniconda instead,
as miniconda's pip doesn't install packages to the system library locations when python is installed.

https://github.com/conda-forge/miniforge/#download

(using miniforge command line window)

 conda create --name fabada --no-default-packages python=3.10
 conda activate fabada
 pip install pipwin, dearpygui, numba, np_rw_buffer,matplotlib, snowy
 pipwin install pyaudio


pythonw.exe thepythonfilename.py #assuming the python file is in the current directory


Usage:
You'll need a line-in device or virtual audio cable you can configure so you can loop the output to input.
The streaming example included here looks for the windows muxer that configures audio devices- whatever you set
in your windows settings for default mic and speaker, respectively, this program will treat as input and output.
So, you can configure another program to output noisy sound to the speaker side of a virtual audio device, and configure
the microphone end of that device as your system microphone, then this program will automatically pick it up and run it.
https://vb-audio.com/Cable/ is an example of a free audio cable.
The program expects 44100hz audio, 16 bit, two channel, but can be configured to work with anything
Additional thanks to Justin Engel.

"""

from __future__ import division

import numpy
import pyaudio
import numba
from matplotlib import mlab
from np_rw_buffer import AudioFramingBuffer
from np_rw_buffer import RingBuffer
from threading import Thread
import math
import time
from time import sleep
import dearpygui.dearpygui as dpg
import snowy
from collections.abc import Mapping, MutableMapping
import numpy as np
from numpy import ma

import matplotlib as mpl
from matplotlib import _api, colors, cbook
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed


@_api.caching_module_getattr  # module-level deprecations
class __getattr__:
    LUTSIZE = _api.deprecated(
        "3.5", obj_type="", alternative="rcParams['image.lut']")(
            property(lambda self: _LUTSIZE))


_LUTSIZE = mpl.rcParams['image.lut']


def _gen_cmap_registry():
    """
    Generate a dict mapping standard colormap names to standard colormaps, as
    well as the reversed colormaps.
    """
    cmap_d = {**cmaps_listed}
    for name, spec in datad.items():
        cmap_d[name] = (  # Precache the cmaps at a fixed lutsize..
            colors.LinearSegmentedColormap(name, spec, _LUTSIZE)
            if 'red' in spec else
            colors.ListedColormap(spec['listed'], name)
            if 'listed' in spec else
            colors.LinearSegmentedColormap.from_list(name, spec, _LUTSIZE))
    # Generate reversed cmaps.
    for cmap in list(cmap_d.values()):
        rmap = cmap.reversed()
        cmap._global = True
        rmap._global = True
        cmap_d[rmap.name] = rmap
    return cmap_d


class _DeprecatedCmapDictWrapper(MutableMapping):
    """Dictionary mapping for deprecated _cmap_d access."""

    def __init__(self, cmap_registry):
        self._cmap_registry = cmap_registry

    def __delitem__(self, key):
        self._warn_deprecated()
        self._cmap_registry.__delitem__(key)

    def __getitem__(self, key):
        self._warn_deprecated()
        return self._cmap_registry.__getitem__(key)

    def __iter__(self):
        self._warn_deprecated()
        return self._cmap_registry.__iter__()

    def __len__(self):
        self._warn_deprecated()
        return self._cmap_registry.__len__()

    def __setitem__(self, key, val):
        self._warn_deprecated()
        self._cmap_registry.__setitem__(key, val)

    def get(self, key, default=None):
        self._warn_deprecated()
        return self._cmap_registry.get(key, default)

    def _warn_deprecated(self):
        _api.warn_deprecated(
            "3.3",
            message="The global colormaps dictionary is no longer "
                    "considered public API.",
            alternative="Please use register_cmap() and get_cmap() to "
                        "access the contents of the dictionary."
        )


class ColormapRegistry(Mapping):

    def __init__(self, cmaps):
        self._cmaps = cmaps

    def __getitem__(self, item):
        try:
            return self._cmaps[item].copy()
        except KeyError:
            raise KeyError(f"{item!r} is not a known colormap name")

    def __iter__(self):
        return iter(self._cmaps)

    def __len__(self):
        return len(self._cmaps)

    def __str__(self):
        return ('ColormapRegistry; available colormaps:\n' +
                ', '.join(f"'{name}'" for name in self))

    def __call__(self):

        return list(self)

    def register(self, cmap, *, name=None, force=False):

        name = name or cmap.name
        if name in self and not force:
            raise ValueError(
                f'A colormap named "{name}" is already registered.')
        register_cmap(name, cmap.copy())


_cmap_registry = _gen_cmap_registry()
globals().update(_cmap_registry)
# This is no longer considered public API
cmap_d = _DeprecatedCmapDictWrapper(_cmap_registry)
__builtin_cmaps = tuple(_cmap_registry)

# public access to the colormaps should be via `matplotlib.colormaps`. For now,
# we still create the registry here, but that should stay an implementation
# detail.
_colormaps = ColormapRegistry(_cmap_registry)


def register_cmap(name=None, cmap=None, *, override_builtin=False):

    _api.check_isinstance((str, None), name=name)
    if name is None:
        try:
            name = cmap.name
        except AttributeError as err:
            raise ValueError("Arguments must include a name or a "
                             "Colormap") from err
    if name in _cmap_registry:
        if not override_builtin and name in __builtin_cmaps:
            msg = f"Trying to re-register the builtin cmap {name!r}."
            raise ValueError(msg)
        else:
            msg = f"Trying to register the cmap {name!r} which already exists."
            _api.warn_external(msg)

    if not isinstance(cmap, colors.Colormap):
        raise ValueError("You must pass a Colormap instance. "
                         f"You passed {cmap} a {type(cmap)} object.")

    cmap._global = True
    _cmap_registry[name] = cmap
    return


def get_cmap(name=None, lut=None):

    if name is None:
        name = mpl.rcParams['image.cmap']
    if isinstance(name, colors.Colormap):
        return name
    _api.check_in_list(sorted(_cmap_registry), name=name)
    if lut is None:
        return _cmap_registry[name]
    else:
        return _cmap_registry[name]._resample(lut)


def unregister_cmap(name):

    if name not in _cmap_registry:
        return
    if name in __builtin_cmaps:
        raise ValueError(f"cannot unregister {name!r} which is a builtin "
                         "colormap.")
    return _cmap_registry.pop(name)


class ScalarMappable:
    """
    A mixin class to map scalar data to RGBA.
    The ScalarMappable applies data normalization before returning RGBA colors
    from the given colormap.
    """

    def __init__(self, norm=None, cmap=None):
        """
        Parameters
        ----------
        norm : `matplotlib.colors.Normalize` (or subclass thereof)
            The normalizing object which scales data, typically into the
            interval ``[0, 1]``.
            If *None*, *norm* defaults to a *colors.Normalize* object which
            initializes its scaling based on the first data processed.
        cmap : str or `~matplotlib.colors.Colormap`
            The colormap used to map normalized data values to RGBA colors.
        """
        self._A = None
        self._norm = None  # So that the setter knows we're initializing.
        self.set_norm(norm)  # The Normalize instance of this ScalarMappable.
        self.cmap = None  # So that the setter knows we're initializing.
        self.set_cmap(cmap)  # The Colormap instance of this ScalarMappable.
        #: The last colorbar associated with this ScalarMappable. May be None.
        self.colorbar = None
        self.callbacks = cbook.CallbackRegistry()

    callbacksSM = _api.deprecated("3.5", alternative="callbacks")(
        property(lambda self: self.callbacks))

    def _scale_norm(self, norm, vmin, vmax):
        """
        Helper for initial scaling.
        Used by public functions that create a ScalarMappable and support
        parameters *vmin*, *vmax* and *norm*. This makes sure that a *norm*
        will take precedence over *vmin*, *vmax*.
        Note that this method does not set the norm.
        """
        if vmin is not None or vmax is not None:
            self.set_clim(vmin, vmax)
            if norm is not None:
                raise ValueError(
                    "Passing parameters norm and vmin/vmax simultaneously is "
                    "not supported. Please pass vmin/vmax directly to the "
                    "norm when creating it.")

        # always resolve the autoscaling so we have concrete limits
        # rather than deferring to draw time.
        self.autoscale_None()

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        """
        Return a normalized rgba array corresponding to *x*.
        In the normal case, *x* is a 1D or 2D sequence of scalars, and
        the corresponding ndarray of rgba values will be returned,
        based on the norm and colormap set for this ScalarMappable.
        There is one special case, for handling images that are already
        rgb or rgba, such as might have been read from an image file.
        If *x* is an ndarray with 3 dimensions,
        and the last dimension is either 3 or 4, then it will be
        treated as an rgb or rgba array, and no mapping will be done.
        The array can be uint8, or it can be floating point with
        values in the 0-1 range; otherwise a ValueError will be raised.
        If it is a masked array, the mask will be ignored.
        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
        will be used to fill in the transparency.  If the last dimension
        is 4, the *alpha* kwarg is ignored; it does not
        replace the pre-existing alpha.  A ValueError will be raised
        if the third dimension is other than 3 or 4.
        In either case, if *bytes* is *False* (default), the rgba
        array will be floats in the 0-1 range; if it is *True*,
        the returned rgba array will be uint8 in the 0 to 255 range.
        If norm is False, no normalization of the input data is
        performed, and it is assumed to be in the range (0-1).
        """
        # First check for special case, image input:
        try:
            if x.ndim == 3:
                if x.shape[2] == 3:
                    if alpha is None:
                        alpha = 1
                    if x.dtype == np.uint8:
                        alpha = np.uint8(alpha * 255)
                    m, n = x.shape[:2]
                    xx = np.empty(shape=(m, n, 4), dtype=x.dtype)
                    xx[:, :, :3] = x
                    xx[:, :, 3] = alpha
                elif x.shape[2] == 4:
                    xx = x
                else:
                    raise ValueError("Third dimension must be 3 or 4")
                if xx.dtype.kind == 'f':
                    if norm and (xx.max() > 1 or xx.min() < 0):
                        raise ValueError("Floating point image RGB values "
                                         "must be in the 0..1 range.")
                    if bytes:
                        xx = (xx * 255).astype(np.uint8)
                elif xx.dtype == np.uint8:
                    if not bytes:
                        xx = xx.astype(np.float32) / 255
                else:
                    raise ValueError("Image RGB array must be uint8 or "
                                     "floating point; found %s" % xx.dtype)
                return xx
        except AttributeError:
            # e.g., x is not an ndarray; so try mapping it
            pass

        # This is the normal case, mapping a scalar array:
        x = ma.asarray(x)
        if norm:
            x = self.norm(x)
        rgba = self.cmap(x, alpha=alpha, bytes=bytes)
        return rgba

    def set_array(self, A):
        """
        Set the value array from array-like *A*.
        Parameters
        ----------
        A : array-like or None
            The values that are mapped to colors.
            The base class `.ScalarMappable` does not make any assumptions on
            the dimensionality and shape of the value array *A*.
        """
        if A is None:
            self._A = None
            return

        A = cbook.safe_masked_invalid(A, copy=True)
        if not np.can_cast(A.dtype, float, "same_kind"):
            raise TypeError(f"Image data of dtype {A.dtype} cannot be "
                            "converted to float")

        self._A = A

    def get_array(self):
        """
        Return the array of values, that are mapped to colors.
        The base class `.ScalarMappable` does not make any assumptions on
        the dimensionality and shape of the array.
        """
        return self._A

    def get_cmap(self):
        """Return the `.Colormap` instance."""
        return self.cmap

    def get_clim(self):
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
        return self.norm.vmin, self.norm.vmax

    def set_clim(self, vmin=None, vmax=None):
        """
        Set the norm limits for image scaling.
        Parameters
        ----------
        vmin, vmax : float
             The limits.
             The limits may also be passed as a tuple (*vmin*, *vmax*) as a
             single positional argument.
             .. ACCEPTS: (vmin: float, vmax: float)
        """
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        if vmax is None:
            try:
                vmin, vmax = vmin
            except (TypeError, ValueError):
                pass
        if vmin is not None:
            self.norm.vmin = colors._sanitize_extrema(vmin)
        if vmax is not None:
            self.norm.vmax = colors._sanitize_extrema(vmax)

    def get_alpha(self):
        """
        Returns
        -------
        float
            Always returns 1.
        """
        # This method is intended to be overridden by Artist sub-classes
        return 1.

    def set_cmap(self, cmap):
        """
        Set the colormap for luminance data.
        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
        in_init = self.cmap is None
        cmap = get_cmap(cmap)
        self.cmap = cmap
        if not in_init:
            self.changed()  # Things are not set up properly yet.

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, norm):
        _api.check_isinstance((colors.Normalize, None), norm=norm)
        if norm is None:
            norm = colors.Normalize()

        if norm is self.norm:
            # We aren't updating anything
            return

        in_init = self.norm is None
        # Remove the current callback and connect to the new one
        if not in_init:
            self.norm.callbacks.disconnect(self._id_norm)
        self._norm = norm
        self._id_norm = self.norm.callbacks.connect('changed',
                                                    self.changed)
        if not in_init:
            self.changed()

    def set_norm(self, norm):
        """
        Set the normalization instance.
        Parameters
        ----------
        norm : `.Normalize` or None
        Notes
        -----
        If there are any colorbars using the mappable for this norm, setting
        the norm of the mappable will reset the norm, locator, and formatters
        on the colorbar to default.
        """
        self.norm = norm

    def autoscale(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self.norm.autoscale(self._A)

    def autoscale_None(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self.norm.autoscale_None(self._A)

    def changed(self):
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal.
        """
        self.callbacks.process('changed', self)
        self.stale = True

@numba.jit(numba.float64[:](numba.float64[:], numba.int32, numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def shift1d(arr : list[numpy.float64], num: int, fill_value: list[numpy.float64]) -> list[numpy.float64] :
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value[:num]
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value[:num]
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

@numba.jit(numba.float64[:,:](numba.float64[:,:], numba.int32, numba.float64[:,:]), nopython=True, parallel=True, nogil=True,cache=True)
def shift2dy(arr: list[numpy.float64], num: int, fill_value: list[numpy.float64]) -> list[numpy.float64] :
    result = np.empty_like(arr)
    if num > 0:
        result[:,:num] = fill_value[:,:num]
        result[:,num:] = arr[:,:-num]
    elif num < 0:
        result[:,num:] = fill_value[:,:num]
        result[:,:num] = arr[:,-num:]
    else:
        result[::] = arr
    return result

@numba.jit(numba.float64[:,:,:](numba.float64[:,:,:], numba.int32, numba.float64[:,:,:]), nopython=True, parallel=True, nogil=True,cache=True)
def shift3dx(arr: list[numpy.float64], num: int, fill_value: list[numpy.float64]) -> list[numpy.float64] :
    result = np.empty_like(arr)
    if num > 0:
        result[:num,:,:] = fill_value[:num,:,:]
        result[num:,:,:] = arr[:-num,:,:]
    elif num < 0:
        result[num:,:,:] = fill_value[:num,:,:]
        result[:num,:,:] = arr[-num:,:,:]
    else:
        result[:] = arr
    return result

@numba.jit(numba.float32[:,:,:](numba.float32[:,:,:], numba.int32, numba.float32[:,:,:]), nopython=True)
def shift3dximg(arr: list[numpy.float32], num: int, fill_value: list[numpy.float32]) -> list[numpy.float32] :
    result = np.empty_like(arr)
    if num > 0:
        result[:,:num,:] = fill_value
        result[:,num:,:] = arr[:,:-num,:]
    elif num < 0:
        result[:,num:,:] = fill_value
        result[:,:num,:] = arr[:,-num:,:]
    else:
        result[:] = arr
    return result

#because numpy's zeroth array is the Y axis, we have to do this in the 1st dimension to shift the X axis
#if the arrays are not the same size, don't attempt to use coordinates for fill value- it will fail.




@numba.jit(numba.types.Tuple((numba.int32, numba.float64[:]))(numba.float64[:], numba.float64, numba.float64,numba.float64), nopython=True, parallel=True, nogil=True,cache=True)
def numba_fabada(data: list[numpy.float64], timex: float, work: float,floor: float) -> (int, list[numpy.float64]):
    # notes:
    # The pythonic way to COPY an array is to do x[:] = y[:]
    # do x=y and it wont copy it, so any changes made to X will also be made to Y.
    # also, += does an append instead of a +
    # math.sqrt is 7x faster than numpy.sqrt but not designed for complex numbers.
    # specifying the type and size in advance of all variables accelerates their use, particularily when JIT is used.
    # However, JIT does not permit arbitrary types like UNION? maybe it does but i havnt figured out how.
    # this implementation uses a lot of for loops because they can be easily vectorized by simply replacing
    # range with numba.prange and also because they will translate well to other languages
    # this implementation of FABADA is not optimized for 2d arrays, however, it is easily swapped by changing the means
    # estimation and by simply changing all other code to iterate over 2d instead of 1d
    # care must be taken with numba parallelization/vectorization

    with numba.objmode(start=numba.float64):
        start = time.time()

    iterations: int = 1
    TAU: float = 2 * math.pi
    N = data.size
    Nf = float(data.size)

    # must establish zeros for the model or otherwise when data is empty, algorithm will return noise
    bayesian_weight = numpy.zeros_like(data)
    bayesian_model = numpy.zeros_like(data)
    model_weight = numpy.zeros_like(data)

    # pre-declaring all arrays allows their memory to be allocated in advance
    posterior_mean = numpy.empty_like(data)
    posterior_variance = numpy.empty_like(data)
    initial_evidence = numpy.empty_like(data)
    evidence = numpy.empty_like(data)
    prior_mean = numpy.empty_like(data)
    prior_variance = numpy.empty_like(data)

    # working set arrays, no real meaning, just to have work space
    ja1 = numpy.empty_like(data)
    ja2 = numpy.empty_like(data)
    ja3 = numpy.empty_like(data)
    ja4 = numpy.empty_like(data)

    # eliminate divide by zero
    data[data == 0.0] = 2.22044604925e-16
    min_d: float = numpy.nanmin(data)
    max_d: float = numpy.amax(data)
    min1: float = 2.22044604925e-16
    max1: float = work
    evidencesum: float = 0.0
    # the higher max is, the less "crackle".
    # The lower floor is, the less noise reduction is possible.
    # floor can never be less than max/2.
    # noise components are generally higher frequency.
    # the higher the floor is set, the more attenuation of both noise and signal.


    for i in numba.prange(N):
        data[i] = (data[i] - min_d) / (max_d - min_d)

    data_mean = numpy.mean(data)  # get the mean
    data_variance = numpy.empty_like(data)
    for i in numba.prange(N):
        data_variance[i] = math.sqrt(numpy.abs(data_mean - (data[i] * max1) + floor) ** 2) #normalize BEFORE variance

    posterior_mean[:] = data[:]
    prior_mean[:] = data[:]

    posterior_variance[:] = data_variance[:]

    for i in numba.prange(N):
        ja1[i] = ((0.0 - math.sqrt(data[i])) ** 2)
        ja2[i] = ((0.0 + data_variance[i]) * 2)
        ja3[i] = math.sqrt(TAU * (0.0 + data_variance[i]))
    for i in numba.prange(N):
        ja4[i] = math.exp(-ja1[i] / ja2[i])
    for i in numba.prange(N):
        evidence[i] = ja4[i] / ja3[i]
    evidence_previous: float = numpy.mean(evidence)
    initial_evidence[:] = evidence[:]

    chi2_data_min: float = 0.0
    for i in numba.prange(N):
        ja1[i] = data[i] - posterior_mean[i]
    for i in numba.prange(N):
        chi2_data_min += ja1[i] ** 2.0 / data_variance[i]
    chi2_pdf_previous: float = 0.0
    chi2_pdf_derivative_previous: float = 0.0

    # do df calculation for chi2 residues
    df = 5
    z = (2. * math.lgamma(df / 2.))

    while 1:

        # GENERATES PRIORS
        prior_mean[:] = posterior_mean[:]

        for i in numba.prange(N - 1):
            prior_mean[i] = prior_mean[i] + posterior_mean[i + 1]
        for i in numba.prange(N - 1):
            prior_mean[i + 1] = prior_mean[i + 1] + posterior_mean[i]
        prior_mean[0] =  prior_mean[0] + (posterior_mean[1] + posterior_mean[2])/2.
        prior_mean[-1] = prior_mean[-1] + (posterior_mean[-2] + posterior_mean[-3]) / 2.

        for i in numba.prange(N):
            prior_mean[i] = prior_mean[i] / 3.


        prior_variance[:] = posterior_variance[:]

        # APPLY BAYES' THEOREM ((b\a)a)\b?

        for i in numba.prange(N):
            posterior_variance[i] = 1. / (1. / data_variance[i] + 1. / prior_variance[i])
        for i in numba.prange(N):
            posterior_mean[i] = (
                        ((prior_mean[i] / prior_variance[i]) + (data[i] / data_variance[i])) * posterior_variance[i])

        # EVALUATE EVIDENCE

        for i in numba.prange(N):
            ja1[i] = ((prior_mean[i] - math.sqrt(data[i])) ** 2)
            ja2[i] = ((prior_variance[i] + data_variance[i]) * 2.)
            ja3[i] = math.sqrt(TAU * (prior_variance[i] + data_variance[i]))
        for i in numba.prange(N):
            ja4[i] = math.exp(-ja1[i] / ja2[i])
        for i in numba.prange(N):
            evidence[i] = ja4[i] / ja3[i]

        # (math.fsum(evidence) / N) Same thing as numpy.mean but slightly more accurate. Replaces both the mean and the sum calls with JIT'd code..
        # may or may not be faster/slower but reduces dependency on external library
        for i in numba.prange(N):
            evidencesum += evidence[i]
        evidence_derivative: float = (evidencesum/ Nf) - evidence_previous
        # EVALUATE CHI2
        chi2_data: float = 0.0
        for i in numba.prange(N):
            chi2_data += math.sqrt((data[i] - posterior_mean[i]) ** 2) / posterior_mean[i] **2

        #chi2 = square root of (actual  - expected )^2  / expected
        #def chi2(x, y, u, q, r, s):
          #  '''Chisquare as a function of data (x, y, and yerr=u), and model
          #  parameters q, r, and s'''
       #     return np.sum((y - f(x, q, r, s)) ** 2 / u ** 2)

        # COMBINE MODELS FOR THE ESTIMATION
        for i in numba.prange(N):
            model_weight[i] = evidence[i] * chi2_data

        for i in numba.prange(N):
            bayesian_weight[i] = bayesian_weight[i] + model_weight[i]
            bayesian_model[i] = bayesian_model[i] + (model_weight[i] * posterior_mean[i])

        # for any data set which is non-trivial. Remember, DF/2 - 1 becomes the exponent! For anything over a few hundred this quickly exceeds float64.
        # chi2.pdf(x, df) = 1 / (2*gamma(df/2)) * (x/2)**(df/2-1) * exp(-x/2)
        gamman: numpy.float64 = (chi2_data / 2.)
        gammas: numpy.float64 = ((gamman) ** z) # TODO
        gammaq: numpy.float64 = math.exp(-chi2_data / 2.)
        # for particularily large values, math.exp just returns 0.0
        # TODO
        chi2_pdf = (1. / z) * gammas * gammaq
        # COMBINE MODELS FOR THE ESTIMATION

        chi2_pdf_derivative = chi2_pdf - chi2_pdf_previous
        chi2_pdf_snd_derivative = chi2_pdf_derivative - chi2_pdf_derivative_previous
        chi2_pdf_previous = chi2_pdf
        chi2_pdf_derivative_previous = chi2_pdf_derivative
        evidence_previous = evidence_derivative

        with numba.objmode(current=numba.float64):
           current = time.time()
        timerun = (current - start) * 1000

        if (
                (chi2_data > data.size and chi2_pdf_snd_derivative >= 0)
                and (evidence_derivative < 0)
                or (int(timerun) > int(timex))  # use no more than the time allocated per cycle
        ):
            break

        iterations += 1

        # COMBINE ITERATION ZERO
    for i in numba.prange(N):
        model_weight[i] = initial_evidence[i] * chi2_data_min
    for i in numba.prange(N):
        bayesian_weight[i] = (bayesian_weight[i] + model_weight[i])
        bayesian_model[i] = bayesian_model[i] + (model_weight[i] * data[i])

    for i in numba.prange(N):
        data[i] = bayesian_model[i] / bayesian_weight[i]
    for i in numba.prange(N):
        data[i] = (data[i] * (max_d - min_d) + min_d)
    return iterations, data

class FilterRun(Thread):
    def __init__(self, rb, pb, channels, processing_size, dtype,fftlow,ffthigh,work,time,floor,iterations,dirty,clean,run):
        super(FilterRun, self).__init__()
        self.running = True
        self.rb = rb
        self.processedrb = pb
        self.channels = channels
        self.processing_size = processing_size
        self.dtype = dtype
        self.buffer = numpy.ndarray(dtype=numpy.float64, shape=[int(self.processing_size * self.channels)])
        self.buffer2 = numpy.ndarray(dtype=numpy.float64, shape=[int(self.processing_size * self.channels)])
        self.buffer2 = self.buffer.reshape(-1, self.channels)
        self.buffer = self.buffer.reshape(-1, self.channels)
        self.fftlow = fftlow
        self.ffthigh = ffthigh
        self.work = work
        self.time = time
        self.floor = floor
        self.iterations = iterations
        self.dirtyspecbuf = dirty
        self.cleanspecbuf = clean
        self.enabled = run
        self.NFFT = 512
        self.noverlap=446
        self.SM = ScalarMappable(cmap="turbo")

    def write_filtered_data(self):
        numpy.copyto(self.buffer, self.rb.read(self.processing_size).astype(dtype=numpy.float64))
        #self.buffer contains (44100,2) of numpy.float32 audio values sampled with pyaudio
        Z, freqs, t = mlab.specgram(self.buffer[:, 0], NFFT=self.NFFT, Fs=44100, detrend=None, window=None, noverlap=self.noverlap,
                                    pad_to=None, sides=None, scale_by_freq=None, mode="magnitude")

        #https://stackoverflow.com/questions/39359693/single-valued-array-to-rgba-array-using-custom-color-map-in-python
        arr_color = self.SM.to_rgba(Z, bytes=False,norm=True)
        arr_color = arr_color[:150, :, :] #we just want the last bits where the specgram data lies.
        arr_color = snowy.resize(arr_color, width=120, height=257)  # in the future, this width will be 60.
        arr_color = numpy.rot90(arr_color)#rotate it and jam it in the buffer lengthwise
        self.dirtyspecbuf.growing_write(arr_color)
        #buffer receives the image translated and on the other end it's reversed and appended a few lines at a time to a texture



        #callbackmain(self.buffer[:, 0],0)
        iterationz = 0
        if self.enabled == False:
            self.processedrb.write(self.buffer.astype(dtype=self.dtype), error=True)
            self.cleanspecbuf.growing_write(arr_color)
            return
        for i in range(self.channels):
            fft = numpy.fft.rfft(self.buffer[:, i])
            zeros = numpy.zeros_like(fft)
            band = numpy.zeros_like(fft)
            band[self.fftlow:self.ffthigh] = fft[self.fftlow:self.ffthigh]
            iteration, band = numba_fabada(numpy.fft.irfft(band), 360.0, self.work, self.floor)
            band = numpy.fft.rfft(band)
            zeros[self.fftlow:self.ffthigh] = band[self.fftlow:self.ffthigh]
            self.buffer2[:, i] = numpy.fft.irfft(zeros)
            iterationz = iterationz + iteration
        self.iterations = iterationz
        self.processedrb.write(self.buffer2.astype(dtype=self.dtype), error=True)

        Z, freqs, t = mlab.specgram(self.buffer2[:, 0], NFFT=512, Fs=44100.0, detrend=None, window=None, noverlap=446,
                                    pad_to=None, scale_by_freq=None, mode="magnitude")

        # https://stackoverflow.com/questions/39359693/single-valued-array-to-rgba-array-using-custom-color-map-in-python
        arr_color = self.SM.to_rgba(Z, bytes=False,norm=True)
        arr_color = arr_color[:150, :, :]  # we just want the last bits where the specgram data lies.
        arr_color = snowy.resize(arr_color, width=120, height=257)  # in the future, this width will be 60.
        arr_color = numpy.rot90(arr_color)#rotate it and jam it in the buffer lengthwise
        self.cleanspecbuf.growing_write(arr_color)

    def run(self):
        while self.running:
            if len(self.rb) < self.processing_size * 2:
                sleep(0.05)  # idk how long we should sleep
            else:
                self.write_filtered_data()

    def stop(self):
        self.running = False


class StreamSampler(object):
    dtype_to_paformat = {
        # Numpy dtype : pyaudio enum
        'uint8': pyaudio.paUInt8,
        'int8': pyaudio.paInt8,
        'uint16': pyaudio.paInt16,
        'int16': pyaudio.paInt16,
        'uint24': pyaudio.paInt24,
        'int24': pyaudio.paInt24,
        "uint32": pyaudio.paInt32,
        'int32': pyaudio.paInt32,
        'float32': pyaudio.paFloat32,

        # Float64 is not a valid pyaudio type.
        # The encode method changes this to a float32 before sending to audio
        'float64': pyaudio.paFloat32,
        "complex128": pyaudio.paFloat32,
    }

    @classmethod
    def get_pa_format(cls, dtype):
        try:
            dtype = dtype.dtype
        except (AttributeError, Exception):
            pass
        return cls.dtype_to_paformat[dtype.name]

    def __init__(self, sample_rate=44100, channels=2, buffer_delay=1.5,  # or 1.5, measured in seconds
                 micindex=1, speakerindex=1, dtype=numpy.float32):
        self.pa = pyaudio.PyAudio()
        self._processing_size = sample_rate
        # np_rw_buffer (AudioFramingBuffer offers a delay time)
        self._sample_rate = sample_rate
        self._channels = channels
        self.ticker = 0
        # self.rb = RingBuffer((int(sample_rate) * 5, channels), dtype=numpy.dtype(dtype))
        self.rb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                     seconds=6,  # Buffer size (need larger than processing size)[seconds * sample_rate]
                                     buffer_delay=0,  # #this buffer doesnt need to have a size
                                     dtype=numpy.dtype(dtype))

        self.processedrb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                              seconds=6,
                                              # Buffer size (need larger than processing size)[seconds * sample_rate]
                                              buffer_delay=0,
                                              # as long as fabada completes in O(n) of less than the sample size in time
                                              dtype=numpy.dtype(dtype))
        self.cleanspectrogrambuffer = RingBuffer((660, 257, 4),dtype=numpy.float32)
        self.dirtyspectrogrambuffer = RingBuffer((660, 257, 4),dtype=numpy.float32)
        self.cleanspectrogrambuffer.maxsize = int(9900)
        self.dirtyspectrogrambuffer.maxsize = int(9900)
        self.texture1 = [1., 1., 1., 1.] * 660 * 257
        self.texture1 = numpy.asarray( self.texture1,dtype=numpy.float32)
        self.texture2 = [1., 1., 1., 1.] * 660 * 257
        self.texture2 = numpy.asarray( self.texture2,dtype=numpy.float32)
        self.texture1 = self.texture1.reshape((257, 660, 4))
        self.texture2 = self.texture2.reshape((257, 660, 4)) #create and shape the textures. Backwards.




        self.fftlow = 20
        self.ffthigh = 11025
        self.work = 1. #included only for completeness
        self.time = 495 #generally, set this to whatever timeframe you want it done in. 44100 samples = 500ms window.
        self.floor = 8192#unknown, seems to do better with higher values
        self.iterations = 0
        self.enabled = True
        self.filterthread = FilterRun(self.rb, self.processedrb, self._channels, self._processing_size, self.dtype,self.fftlow,self.ffthigh,self.work,self.time, self.floor,self.iterations,self.dirtyspectrogrambuffer,self.cleanspectrogrambuffer,self.enabled)
        self.micindex = micindex
        self.speakerindex = speakerindex
        self.micstream = None
        self.speakerstream = None
        self.speakerdevice = ""
        self.micdevice = ""


        # Set inputs for inheritance
        self.set_sample_rate(sample_rate)
        self.set_channels(channels)
        self.set_dtype(dtype)

    @property
    def processing_size(self):
        return self._processing_size

    @processing_size.setter
    def processing_size(self, value):
        self._processing_size = value
        self._update_streams()

    def get_sample_rate(self):
        return self._sample_rate

    def set_sample_rate(self, value):
        self._sample_rate = value
        try:  # RingBuffer
            self.rb.maxsize = int(value * 5)
            self.processedrb.maxsize = int(value * 5)
        except AttributeError:
            pass
        try:  # AudioFramingBuffer
            self.rb.sample_rate = value
            self.processedrb.sample_rate = value
        except AttributeError:
            pass
        self._update_streams()

    sample_rate = property(get_sample_rate, set_sample_rate)

    def get_channels(self):
        return self._channels

    def set_channels(self, value):
        self._channels = value
        try:  # RingBuffer
            self.rb.columns = value
            self.processedrb.columns = value
        except AttributeError:
            pass
        try:  # AudioFrammingBuffer
            self.rb.channels = value
            self.processedrb.channels = value
        except AttributeError:
            pass
        self._update_streams()

    channels = property(get_channels, set_channels)

    def get_dtype(self):
        return self.rb.dtype

    def set_dtype(self, value):
        try:
            self.rb.dtype = value
        except AttributeError:
            pass
        self._update_streams()

    dtype = property(get_dtype, set_dtype)

    @property
    def pa_format(self):
        return self.get_pa_format(self.dtype)

    @pa_format.setter
    def pa_format(self, value):
        for np_dtype, pa_fmt in self.dtype_to_paformat.items():
            if value == pa_fmt:
                self.dtype = numpy.dtype(np_dtype)
                return

        raise ValueError('Invalid pyaudio format given!')

    @property
    def buffer_delay(self):
        try:
            return self.rb.buffer_delay
        except (AttributeError, Exception):
            return 0

    @buffer_delay.setter
    def buffer_delay(self, value):
        try:
            self.rb.buffer_delay = value
            self.processedrb.buffer_delay = value
        except AttributeError:
            pass

    def _update_streams(self):
        """Call if sample rate, channels, dtype, or something about the stream changes."""
        was_running = self.is_running()

        self.stop()
        self.micstream = None
        self.speakerstream = None
        if was_running:
            self.listen()

    def is_running(self):
        try:
            return self.micstream.is_active() or self.speakerstream.is_active()
        except (AttributeError, Exception):
            return False

    def stop(self):
        try:
            self.micstream.close()
        except (AttributeError, Exception):
            pass
        try:
            self.speakerstream.close()
        except (AttributeError, Exception):
            pass
        try:
            self.filterthread.join()
        except (AttributeError, Exception):
            pass

    def open_mic_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            # print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxInputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        self.micdevice = devinfo["name"]
                        device_index = i
                        self.micindex = device_index

        if device_index is None:
            print("No preferred input found; using default input device.")

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              input=True,
                              input_device_index=self.micindex,  # device_index,
                              # each frame carries twice the data of the frames
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_read,
                              start=False  # Need start to be False if you don't want this to start right away
                              )

        return stream

    def open_speaker_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            # print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxOutputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        self.speakerdevice = devinfo["name"]
                        device_index = i
                        self.speakerindex = device_index

        if device_index is None:
            print("No preferred output found; using default output device.")

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              output=True,
                              output_device_index=self.speakerindex,
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_write,
                              start=False  # Need start to be False if you don't want this to start right away
                              )
        return stream

    # it is critical that this function do as little as possible, as fast as possible. numpy.ndarray is the fastest we can move.
    # attention: numpy.ndarray is actually faster than frombuffer for known buffer sizes
    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        audio_in = memoryview(numpy.ndarray(buffer=memoryview(in_data), dtype=self.dtype,
                                            shape=[int(self._processing_size * self._channels)]).reshape(-1,
                                                                                                         self.channels))
        self.rb.write(audio_in, error=False)
        return None, pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
        # Read raw data
        # filtered = self.rb.read(frame_count)
        # if len(filtered) < frame_count:
        #     filtered = numpy.zeros((frame_count, self.channels), dtype=self.dtype)
        if len(self.processedrb) < self.processing_size:
            # print('Not enough data to play! Increase the buffer_delay')
            # uncomment this for debug
            audio = numpy.zeros((self.processing_size, self.channels), dtype=self.dtype)
            return audio, pyaudio.paContinue

        audio = self.processedrb.read(self.processing_size)
        chans = []
        for i in range(self.channels):
            filtered = audio[:, i]
            chans.append(filtered)

        return numpy.column_stack(chans).astype(self.dtype).tobytes(), pyaudio.paContinue



    def stream_start(self):
        if self.micstream is None:
            self.micstream = self.open_mic_stream()
        self.micstream.start_stream()

        if self.speakerstream is None:
            self.speakerstream = self.open_speaker_stream()
        self.speakerstream.start_stream()
        # Don't do this here. Do it in main. Other things may want to run while this stream is running
        # while self.micstream.is_active():
        #     eval(input("main thread is now paused"))

    listen = stream_start  # Just set a new variable to the same method



if __name__ == "__main__":
    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all cores
    SS = StreamSampler(buffer_delay=0)
    SS.listen()
    SS.filterthread.start()
    def close():
        dpg.destroy_context()
        SS.filterthread.stop()
        SS.stop()
        quit()

    def fftminlog(sender, app_data):
        value = int(app_data)
        if value < SS.ffthigh:
            SS.fftlow = value
        else:
            dpg.set_value('FFTmin',SS.ffthigh-1)
            SS.fftlow = SS.ffthigh-1
        SS.filterthread.fftlow = SS.fftlow


    def fftmaxlog(sender, app_data):
        value = int(app_data)
        if value > SS.fftlow:
            SS.ffthigh = value
        else:
            dpg.set_value('FFTMax',SS.fftlow+1)
            SS.ffthigh = SS.fftlow+1
        SS.filterthread.ffthigh = SS.ffthigh


    def update_spectrogram_textures():
        #new_color = implement buffer read
        if len(SS.dirtyspectrogrambuffer) < 660 * 2 or len(SS.cleanspectrogrambuffer) < 660 * 2:
            return
        dirty_data = SS.dirtyspectrogrambuffer.read(2)
        dirty_data = numpy.rot90(dirty_data,1)#pull it out and flop it on the table
        SS.texture1 = shift3dximg(SS.texture1,-2,dirty_data)
        #SS.texture1 = numpy.roll(SS.texture1, -2, axis=1)#shift the contents of this array by -2
        #SS.texture1[:,-2:,:] = dirty_data #fill the last two rows with the dirty data
        clean_data = SS.cleanspectrogrambuffer.read(2)
        clean_data = numpy.rot90(clean_data, 1)  # pull it out and flop it on the table
        SS.texture2 = shift3dximg(SS.texture2, -2, clean_data)
        #dirty_data =  cv2.flip(SS.dirtyspectrogrambuffer.read_overlap(682,4),0)#read a given amount and overlap the next read
        #clean_data =  cv2.flip(SS.cleanspectrogrambuffer.read_overlap(682,4),0)#i dont know how much to overlap by.
        dpg.set_value("dirty_texture", SS.texture1)                  #perhaps updating once a frame is too often?
        dpg.set_value("clean_texture", SS.texture2)



    def iter():
        dpg.set_value('iterations', f"Fabada current iterations: {SS.filterthread.iterations}")
        update_spectrogram_textures() #update the screen contents once every 4 frames when using SS.Ticker.

    def fabadatoggle(sender, app_data,user_data):
        if SS.filterthread.enabled == True:
            dpg.set_item_label("toggleswitch", "Enable")
            SS.filterthread.enabled = False
        else:
            dpg.set_item_label("toggleswitch", "Disable")
            SS.filterthread.enabled = True


#notes:
#future version will include two spectrograms.
# https://stackoverflow.com/questions/6800984/how-to-pass-and-run-a-callback-method-in-python
#each time we process a chunk, we'll also copy out the input and output from the thread to a callback function.
#that callback function will be in the main SS program where the spectrograms will be generated and appended to an RGBA ringbuffer.
#https://github.com/alakise/Audio-Spectrogram/blob/master/spectrogram.py
#We will use an overlapping read to "scroll" the video content.
#each time, we'll attempt to copy out what remains in the buffer from X point, and then increment X point.
#if there isn't enough data, we'll just pause the feed and display a text message " samples are being dropped!"
#below the textures. filling with "black" would be nice but will be more complex so we wont be doing that.
#important variables to determine are the canvas height and width of the plot.
#the height will tell us how high to make our texture. The width won't tell us how wide to make it-
#but it will tell us how large the buffer needs to be.
    dpg.create_context()
    dpg.create_viewport(title='FABADA Streamclean', height=700, width=700)
    dpg.setup_dearpygui()
    dpg.configure_app(auto_device=True)

    cleantexture = [1, 1, 0, 1] * 660 * 257
    dirtytexture = [1, 0, 1, 1] * 660 * 257
    #patch from joviex- the enumeration in the online docs showing .append doesn't work for larger textures

    with dpg.texture_registry():
        dpg.add_dynamic_texture(660, 257, dirtytexture, tag="dirty_texture")
    with dpg.texture_registry():
        dpg.add_dynamic_texture(660, 257, cleantexture, tag="clean_texture")


    with dpg.window(autosize=True, width = 700) as main_window:
        dpg.add_text("Welcome to FABADA! Feel free to experiment.")
        dpg.add_text(f"Your speaker device is: ({SS.speakerdevice})")
        dpg.add_text(f"Your microphone device is:({SS.micdevice})")
        dpg.add_text("Fabada current iterations: 0",tag="iterations")
        dpg.add_button(label="Disable", tag="toggleswitch", callback=fabadatoggle)
        dpg.add_slider_int(label="FFTmin", tag="FFTmin", default_value=SS.fftlow, min_value=1,
                           max_value=11025,
                           callback=fftminlog)
        dpg.add_slider_int(label="FFTMax", tag="FFTMax", default_value=SS.ffthigh, min_value=1,
                           max_value=11025,
                           callback=fftmaxlog)

        dpg.add_image("dirty_texture")
        dpg.add_image("clean_texture")

    dpg.set_primary_window(main_window,True)  # TODO: Added Primary window, so the dpg window fills the whole Viewport

    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        iter()#this runs once a frame.
        dpg.render_dearpygui_frame()
    close() #clean up the program runtime when the user closes the window
