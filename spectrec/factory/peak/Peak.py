# -- Import built-in modules
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Sequence, Union, Optional
import warnings

# -- Import third party modules
import numpy as np
import torch


class Peak(metaclass=ABCMeta):


    def __init__(self, limits: dict[str, Sequence[float]], peak_id: Optional[str] = None):
        """ Initialise a Peak using **kwargs to pass the limits. """

        # -- Dictionaries to hold the parameter values and the limits
        self._param_values, self._param_limits = {}, {}

        # Iterate for each parameters defining the peak
        for param_name in self._param_ids:

            # Select the limits of the given parameter
            param_limits = self.__select_param_lims(limits, param_name, peak_id)

            # Assert the limits are correct; two values are passed
            assert len(param_limits) == 2, f'Lower and upper limits for {param_name =} must be provided.'

            # Save the limits in the corresponding dictionary
            self._param_limits[param_name] = sorted(param_limits)

            # Calculate the parameter using the limits
            self._param_values[param_name] = np.random.uniform(param_limits[0], param_limits[1])

    def recalculate(self) -> None:
        """ Randomly generate new parameter values using the same limits. """
        for param_name in self._param_values.keys():
            self._param_values[param_name] = np.random.uniform(
                self._param_limits[param_name][0], self._param_limits[param_name][1]
            )

    def set_parameter_limit(self, param_name: str, limits: list[float]):
        """ Set the limits for the given parameter. Do not forget to recalculate the
        values after setting the new limits

        --- Parameters:
        param_name: str
            Parameter whose limits need to be changed.
        limits: Sequence[float]
            New limits of the parameter
        """

        # Assert some conditions on the input
        assert param_name in self._param_ids, '{param_name =} should be a parameter listed in {self._param_ids}.'
        assert len(limits) == 2, 'Lower and upper limits must be provided.'

        # Set the correct limits
        self._param_limits[param_name] = limits
    
    # -- Private methods of the class {{{
    def __select_param_lims(self, limits: dict[str, Sequence[float]], param_name: str, peak_id: Union[str, None]) -> Sequence[float]:
        """ Select the correct parameter limits from the dictionary of limits. The
        search is carried out using different naming conventions and in a hierarchically
        order:

                    First: param_name + peak_id; second: param_name

        --- Parameters:
        limits: dict
            Dictionary containing several parameter names as keys and their limits as values.
        param_name: str
            Conventional name of the parameter.
        peak_id: str
            Peak id used to identify parameter limits for this current Peak. The parameters corresponding
            to a given peak_id should have the following name convention: param_name + peak_id
        """

        # Get the particular identifier of this peak, if possible
        particular_name = param_name + peak_id if peak_id else param_name

        # Get all keys from the dictionary to process them
        all_names = list(limits.keys())

        # Check whether name_id is in the list of keys
        if peak_id is not None and particular_name not in all_names:
            warnings.warn(f'{peak_id =} specifier not in {all_names}. Using {param_name =} instead.')

        # Assert that at least, one valid name is present in the keys
        assert any(k in all_names for k in (particular_name, param_name)), \
            f'Parameter keys {particular_name}/{param_name} not found in {all_names}'

        # Select the correct key hierarchycally, from more specific to less
        for possible_name in (particular_name, param_name):
            if possible_name in all_names:
                return limits[possible_name]

        return []
    # -- }}}

    # -- Abstract methods of the base class {{{
    @abstractmethod
    def generate(self, omega: torch.Tensor) -> torch.Tensor:
        """ Generate a torch representation of the peak in the given energy range. """
        return
    # -- }}}

    # -- Abstract properties of the base class {{{
    @abstractproperty
    def _param_ids(self) -> Sequence[str]:
        """ Names of the parameters used in the class. For example, [A, M, W] in a Gaussian
        peak with parameters: amplitude (A), mass (M) and width (W).
        """
        return
    # -- }}}

    # -- Magic methods of the class {{{
    def __str__(self) -> str:
        """ String representation of the object. """
        par_str = {k: round(v, 2) for k, v in self.params.items()}
        return f'<{self._peak_type}: {par_str}'

    def __repr__(self) -> str:
        ''' String representation of the object. '''
        return self.__str__()
    # -- }}}

    # -- Property methods of the class {{{
    @property
    def params(self) -> dict[str, float]:
        ''' Return the parameters that define the peak. '''
        return self._param_values

    @property
    def limits(self) -> dict[str, list[float]]:
        """ Return the limits used to define the peak values. """
        return self._param_limits

    @property
    def _peak_type(self) -> str:
        """ Name identifier of the Peak type. """
        return 'Peak'
    # -- }}}


if __name__ == '__main__':
    pass
