from typing import Union

# Import user-defined modules
from spectrec.factory import Peak
from spectrec.factory import Kernel
from spectrec.network import Network

from spectrec.factory import GaussianPeak
from spectrec.factory import DeltaPeak
from spectrec.factory import NRQCDKernel
from spectrec.network import UNet

class NPKFactory:

    # All instances of the class share these dictionaries
    __registrations = {}

    def __init__(self):
        """ The init method registers some classes automatically """

        self.register_class('GaussianPeak', GaussianPeak)
        self.register_class('DeltaPeak',    DeltaPeak)
        self.register_class('NRQCDKernel',  NRQCDKernel)
        self.register_class('UNet',         UNet)

    def register_class(self, name: str, to_register: Union[Peak, Kernel, Network]):
        """ Register a class with a given name in the factory. """

        # Assert the class to register is correct
        assert issubclass(to_register, (Peak, Kernel, Network)), \
            f'{to_register=} must be a subclass of [Peak, Kernel, Network]'

        # Get the subclass of to_register
        sub_class = to_register.__bases__

        # Register the subclass
        self.__registrations[name] = to_register

    def retrieve_class(self, name: str) -> Union[Peak, Kernel, Network]:
        try:
            return self.__registrations[name]
        except KeyError:
            raise KeyError(f'{name=} is not registered: {self.registered_names}')

    def retrieve_all_of_same_type(self, type_name: str) -> dict[str, Union[Peak, Kernel, Network]]:
        """ Get all registered items of the same type, where the type is defined by
        the parent class they inherit from. For example, if type_name is 'Network', then
        it will retrieve all the registered items that are childs of 'Network'. """

        assert type_name.lower() in ['peak', 'kernel', 'network'], \
            f"{type_name=} must be one of ['Peak', 'Kernel', 'Network']"

        # Dictionary that will contain all the registered classes with same type
        same_type = {}

        # Iterate for all registered items
        for name, reg_class in self.__registrations.items():
            if type_name.lower() in str(reg_class.__base__).lower():
                same_type[name] = reg_class

        return same_type

    @property
    def registered_names(self) -> list[str]:
        return list(self.__registrations.keys())

if __name__ == '__main__':
    pass
