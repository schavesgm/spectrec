# Import user-defined modules
from spectrec.factory import Peak
from spectrec.factory import Kernel

# -- Factory to deal with peaks {{{
__registered_peaks = {}

def register_peak_class(name: str, peak: Peak):
    """ Register a peak to be used by the parser. """
    assert issubclass(peak, Peak), '{peak =} must be a subclass of Peak.'
    __registered_peaks[name] = peak

def create_peak_class(name: str):
    """ Return a peak class from the registered set. """
    assert name in __registered_peaks, f'{name =} is not a registered peak.'
    return __registered_peaks[name]
# -- }}}

# -- Factory to deal with kernels {{{
__registered_kernels = {}

def register_kernel_class(name: str, kernel: Kernel):
    """ Register a kernel to be used by the parser. """
    assert issubclass(kernel, Kernel), '{kernel =} must be a subclass of Kernel.'
    __registered_kernels[name] = kernel

def create_kernel_class(name: str):
    """ Return a kernel class from the registered set. """
    assert name in __registered_kernels, f'{name =} is not a registered kernel.'
    return __registered_kernels[name]
# -- }}}

if __name__ == '__main__':
    pass
