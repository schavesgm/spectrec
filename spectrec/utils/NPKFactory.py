# Import user-defined modules
from spectrec.factory import Peak
from spectrec.factory import Kernel
from spectrec.network import Network

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

# -- Factory to deal with networks {{{
__registered_networks = {}

def register_network_class(name: str, network: Network):
    """ Register a network to be used by the parser. """
    assert issubclass(network, Network), f'{network =} must be a subclass of Network.'
    __registered_networks[name] = network

def retrieve_network_class(name: str) -> Network:
    """ Return a network class from the registered set. """
    try:
        return __registered_networks[name]
    except KeyError:
        raise KeyError(f'{name =} is not registered: {__registered_networks.keys()}')
# -- }}}

if __name__ == '__main__':
    pass
