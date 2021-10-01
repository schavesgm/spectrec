# Load some needed classes
from .SpectrecInput import SpectrecInput

# Load the factory functions
from .NPKFactory import register_peak_class
from .NPKFactory import register_kernel_class
from .NPKFactory import register_network_class
from .NPKFactory import retrieve_peak_class
from .NPKFactory import retrieve_kernel_class
from .NPKFactory import retrieve_network_class

# Load some needed functions
from .train import train_network
from .plots import eliminate_mirror_axis
