# Import user-defined modules
from spectrec.factory import Peak
from spectrec.factory import Kernel

# -- Factory to deal with peaks {{{
__registered_peaks = {}

def register_peak_class(name: str, peak: Peak):
    """ Register a peak to be used by the parser. """
    assert issubclass(peak, Peak), f'{peak =} must be a subclass of Peak.'
    __registered_peaks[name] = peak

def retrieve_peak_class(name: str) -> Peak:
    """ Return a peak class from the registered set. """
    try:
        return __registered_peaks[name]
    except KeyError:
        raise KeyError(f'{name =} is not registered: {__registered_peaks.keys()}')
# -- }}}

# -- Factory to deal with kernels {{{
__registered_kernels = {}

def register_kernel_class(name: str, kernel: Kernel):
    """ Register a kernel to be used by the parser. """
    assert issubclass(kernel, Kernel), f'{kernel =} must be a subclass of Kernel.'
    __registered_kernels[name] = kernel

def retrieve_kernel_class(name: str) -> Kernel:
    """ Return a kernel class from the registered set. """
    try:
        return __registered_kernels[name]
    except KeyError:
        raise KeyError(f'{name =} is not registered: {__registered_kernels.keys()}')
# -- }}}

if __name__ == '__main__':
    pass
