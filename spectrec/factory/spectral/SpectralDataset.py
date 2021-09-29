# -- Load some built-in modules
from typing import Optional, Sequence
from collections import namedtuple
import json, os, datetime

# -- Import some third party modules
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.legend_handler import HandlerTuple
 
# -- Import some user-defined modules
from spectrec.factory  import Kernel
from spectrec.factory  import Peak
from .SpectralFunction import SpectralFunction

# -- Spectral function output tuple
SpectralOut = namedtuple('SpectralOut', 'C L')


class SpectralDataset(torch.utils.data.Dataset):
    """ Torch dataset containing spectral functions tranining sets. """

    def __init__(
        self, peak_types: list[Peak], peak_limits: dict[str, Sequence[float]],
        kernel: Kernel, max_np: int, fixed_np: bool = True, peak_ids: Optional[list] = None
    ):

        # Save the peak types and the limits used in the calculation
        self.peak_types, self.peak_limits = peak_types, peak_limits

        # Save the maximum number of peaks and whether they are fixed or not
        self.max_np, self.fixed_np = max_np, fixed_np

        # Save the kernel in the object
        self.kernel = kernel

        # Save the peak_ids depending of the input passed
        if peak_ids:
            # If peak_ids is passed, then it should have the same dimensions as peak_types
            assert len(peak_ids) == len(peak_types), 'There must be a peak if for each peak type'

            # Save the peak_id in the class
            self.peak_ids = peak_ids
        else:
            # If it is not passed, then peak_ids is s list of None
            self.peak_ids = [None] * len(peak_types)

        # Container for the data in the dataset
        self.__data = {'R': None, 'C': None, 'U': None, 'L': None}

    def generate(self, Nb: int, Ns: int, basis: Optional[torch.Tensor] = None, use_GPU: bool = False):

        # Assert the number of examples is greater or equal to Nw
        assert Nb >= self.kernel.Nw, f'{Nb = } must be greater than {self.kernel.Nw = }'

        # If basis functions are passed to the method, assert some conditions on its dimensions
        if basis is not None:
            assert basis.shape == (self.kernel.Nw, Ns), \
                f'Basis dimensions must be = {basis.shape = } != ({self.kernel.Nw =}, {Ns = })'

        # Allocate enough memory for the spectral functions and correlation functions
        self.__data['R'] = torch.empty(Nb, self.kernel.Nw, dtype=torch.float32)
        self.__data['C'] = torch.empty(Nb, self.kernel.Nt, dtype=torch.float32)

        # Generate a spectral function object to sample the data
        spf_gen = SpectralFunction(
            self.peak_types, self.peak_limits, self.kernel, self.max_np, self.fixed_np, self.peak_ids
        )

        # Generate Nb different spectral functions
        for nb in range(Nb):

            # Sample some peaks from the spectral function
            spf_gen.generate_random_peaks()

            # Get the tensor representation of R and C in the correlation function
            self.__data['R'][nb, :] = spf_gen.compute_R(recalculate=True)
            self.__data['C'][nb, :] = spf_gen.compute_C(recalculate=True)

        # Compute the basis functions if basis is not provided or basis is not loaded
        if basis is not None:
            self.__data['U'] = basis
        elif basis is None and self.__data['U'] is None:
            self.__data['U'] = self.__generate_basis(Ns, use_GPU)

        # Get the set of coefficients using the basis functions and the spectral function
        self.__data['L'] = self.__generate_coeffs(use_GPU)

    def clear(self):
        """ Clear all data in the dataset. Sets the data dictionary as None """
        del self.__data['R'], self.__data['C'], self.__data['L'], self.__data['U']
        self.__data = {'R': None, 'C': None, 'U': None, 'L': None}

    def test(
        self, network: torch.nn.Module, loss: torch.nn.Module, prop: float, device: torch.device, 
        batch_size: int, val_or_test: str, prefix: str = '', suffix: str = '', rand: bool = True, 
        path: str = './status/monitor'
    ) -> dict:

        # Assert the dataset has been generated
        assert self.is_generated, 'Dataset cannot be tested as data is not generated.'

        # Assert val_or_test is either val or test
        assert val_or_test in ['val', 'test'], '{val_or_test = } must be in [val, test]'

        # Number of examples in the test set
        Nb_test = int(prop * self.Nb)

        # Get the network in evaluation mode. TODO: Check output Confidence Intervals
        net_test = network.to(device).eval()

        # Select a subset of the data
        test_data = self.__select_prop(prop, rand)

        # Split the test indices into difference batches
        batch_idx = torch.split(torch.arange(0, Nb_test), batch_size)

        # Value of the loss function in the test set
        loss_value = 0.0

        # Buffer containing the predictions of the network
        buffer_pred = torch.empty((Nb_test, self.Ns), dtype = torch.float32)

        # Calculate the network prediction for each batch
        for batch in batch_idx:
            
            # Get the current batch from the set to test the results
            C_batch = test_data['C'][batch, :].detach().to(device).log()
            L_batch = test_data['L'][batch, :].detach().to(device)

            # Generate the correct dimensions of the network
            C_batch = C_batch.view(C_batch.shape[0], 1, C_batch.shape[1])
            L_batch = L_batch.view(L_batch.shape[0], 1, L_batch.shape[1])

            # Calculate the prediction of the network
            L_pred = net_test(C_batch).detach()

            # Calculate the loss function value
            loss_value += float(loss(L_pred, L_batch, C_batch).detach())

            # Save the predicted coefficients of the network
            buffer_pred[batch, :] = L_pred.cpu().view(len(batch), self.Ns)

            # Delete the uneeded tensors
            del C_batch, L_batch, L_pred

        # Add the prefix and the suffix to the name identifier
        identifier = f'{prefix}_{self.name}'  if prefix != '' else self.name
        identifier = f'{identifier}_{suffix}' if suffix != '' else identifier

        # Path where the data will be stored
        path = os.path.join(path, identifier, val_or_test)

        # Create the folder if it does not exist
        if not os.path.exists(path): os.makedirs(path)

        # Dictionary containing information to monitor the test_set TODO: Change
        json_info = self.info | {
            'test': {
                'Nb':   Nb_test,
                'prop': prop,
                'loss': float(loss_value / Nb_test),
                'dL':   float(torch.sum((test_data['L'] - buffer_pred) ** 2, axis=1).mean()),
                'dR':   float((test_data['R'] - buffer_pred @ self.U.T).abs().max(axis=1).values.mean()),
            },
        }

        # Dump the information into the json file
        with open(os.path.join(path, 'json_out.dat'), 'w', encoding='utf8') as json_out:
            json.dump(json_info, json_out, ensure_ascii=False, indent=4)

        # Plot several figures to monitor the behaviour of the network
        for ex in range(4):
            self.__plot_test_examples(buffer_pred, test_data, examples=1).savefig(
                os.path.join(path, f'test_example_{ex}.pdf')
            )

        return json_info

    def save_dataset(self, prefix: str = '', suffix: str = '', path: str = './status/dataset') -> None:
        """ Save all the dataset in a given path. A prefix and a suffix can be appended to the
        name locator to further specify the dataset. The name convention to define a dataset is
        the following:

                        {prefix}_b{Nb}_w{Nw}_s{Ns}_t{Nt}_{suffix}

        --- Parameters:
        prefix: str = ''
            Prefix to be added to the name identifier.
        suffix: str = ''
            Suffix to be added to the name identifier.
        path: str = './status/dataset'
            Output path where the dataset will be saved.

        --- Returns:
        None
        """
        # Data should be generated to be saved
        if not self.is_generated: raise ValueError("Dataset is empty. Generate it.")

        # Add the prefix and the suffix to the name identifier
        identifier = f'{prefix}_{self.name}'  if prefix != '' else self.name
        identifier = f'{identifier}_{suffix}' if suffix != '' else identifier

        # Output folder where the data will be stored
        output_path = os.path.join(path, identifier)

        # Create the ouptut folder if it does not exist
        if not os.path.exists(output_path): os.makedirs(output_path)

        # Save several tensors into the current path
        for name, tensor in self.__data.items():
            torch.save(tensor, os.path.join(output_path, name + '_' + identifier + '.dat'))

        # Save the information as a json output to further specify the dataset
        with open(os.path.join(output_path, 'info.json'), 'w') as f:
            json.dump(self.info, f, indent=4)

    def load_dataset(self, Nb: int, Ns: int, prefix: str = '', suffix: str = '', path: str = './status/dataset') -> None:
        """ Load the dataset from a given path. A prefix and a suffix can be appended to the
        name locator to further specify the dataset. The name convention to define a dataset is
        the following:

                        {prefix}_b{Nb}_w{Nw}_s{Ns}_t{Nt}_{suffix}

        --- Parameters:
        Nb: int
            Select the dataset with Nb examples.
        Ns: int
            Select the dataset with Ns basis functions.
        prefix: str = ''
            Prefix to be added to the name identifier.
        suffix: str = ''
            Suffix to be added to the name identifier.
        path: str = './status/dataset'
            Output path where the dataset will be saved.

        --- Returns:
        None
        """
        # Generate the identifier of the dataset
        identifier = self.__generate_identifier(Nb, Ns, prefix, suffix)

        # Path to the data
        input_path = os.path.join(path, identifier)

        # Check the data is present in the path
        assert os.path.exists(input_path)

        # Assert the dataset to load is correct
        self.__check_correct_dataset(input_path)

        # Load some tensors from the file
        for tensor_id in ['R', 'C', 'L', 'U']:
            self.__load_tensor(tensor_id, identifier, input_path)

    # -- Private methods of the class {{{
    def __generate_basis(self, Ns: int, use_GPU: bool) -> torch.Tensor:
        """ Generate the basis functions using the SVD decomposition of the spectral
        function data. The newly produced basis functions is a dataset of Nw rows and
        Ns functions. The larger the value of Ns, the more precise the approximation
        and the more coefficients we need to fit.

        --- Parameters:
        Ns: int
            Number of basis functions to be extracted.
        use_GPU: bool
            Use GPU-accelerated functions to calculate the SVD. Be careful with memory
            problems.
        
        --- Returns
        torch.Tensor:
            Tensor containing the basis functions.
        """
        # Move the spectral function to the correct domain if use_GPU is passed
        R_temp = self.__data['R'].to('cuda' if use_GPU else 'cpu')

        # Calculate the SVD of the spectral function
        U = torch.linalg.svd(R_temp, full_matrices=False)[2].T[:,:Ns]

        # Delete the uneeded tensor to free up some memory
        del R_temp

        # Return the CPU allocated version of the basis functions
        return U.cpu()

    def __generate_coeffs(self, use_GPU: bool) -> torch.Tensor:
        r""" Generate the coefficients used to expand the spectral functions as a linear
        combination of a set of basis functions. The coefficients would lead to an 
        approximation of the spectral function as an infinite amount of basis functions
        are required. We compute the coefficients by noting,

                    R(w) = \sum_{s=1}^{Ns} l_s \cdot U_s(w),

        thus,
                            l_s = <R(w), U_s(w)>,

        where <.,.> is the dot product.

        --- Parameters:
        use_GPU: bool
            Use GPU-accelerated functions to calculate the coefficients. Be careful with memory
            problems.
        
        --- Returns
        torch.Tensor:
            Tensor containing the coefficients of the expansion.
        """

        # Move R and U to the GPU if the GPU acceleration is on
        R_temp = self.__data['R'].to('cuda' if use_GPU else 'cpu')
        U_temp = self.__data['U'].to('cuda' if use_GPU else 'cpu')

        # Calculate the coefficients using the dot product
        L = R_temp @ U_temp

        # Force the deletion of the uneeded tensors
        del R_temp, U_temp

        # Return the CPU allocated version of the basis functions
        return L.cpu()

    def __select_prop(self, prop: float, rand: bool) -> dict[str, torch.Tensor]:
        """ Select a proportion of the whole dataset randomising it or not. """

        # Number of points to retrieve using perc
        Nb_prop = int(prop * self.Nb)

        # Indices of each example present in the dataset
        idx = torch.arange(0, self.Nb)

        # Select some random values or the first Nb results depending on selection
        idx = torch.randperm(len(idx))[:Nb_prop] if rand else idx[:Nb_prop]
        
        # Fill the dictionary with the correct results
        return {
            'R': self.R[idx, :], 'C': self.C[idx, :], 
            'L': self.L[idx, :], 'U': self.U
        }

    def __plot_test_examples(self, Lp: torch.Tensor, data: dict, examples: int = 1) -> plt.Figure:
        """ Plot some random examples from the test set into a matplotlib's figure. """

        # Color palette used in the plots
        COLORS = ['#168AAD', '#D62828', '#023E8A', '#EE6C4D']

        # Generate the matplotlib figure
        fig = plt.figure(figsize=(16, 10))

        # Generate three (2 left 1 right) axes in the figure
        axis_L1, axis_L2 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 3)
        axis_R           = fig.add_subplot(1, 2, 2)

        # Set some properties in each of the axes
        axis_L1.set_xlabel(r'$n_s$')
        axis_L1.set_ylabel(r'$L(n_s)$')
        axis_L2.set_xlabel(r'$n_s$')
        axis_L2.set_ylabel(r'$|(\hat{L}(n_s) - L(n_s))/L(n_s)|$')
        axis_R.set_xlabel(r'$\omega$')
        axis_R.set_ylabel(r'$\rho(\omega)$')
        axis_L1.grid('#fae1dd', alpha=0.3)
        axis_L2.grid('#fae1dd', alpha=0.3)
        axis_R.grid('#fae1dd', alpha=0.3)

        # Reconstruct the predicted spectral functions from the coefficients
        Rp = (Lp @ self.U.T)

        #List of handlers and labels to customise the legend 
        handles, labels = [], [] 

        # ns values to use in the plots
        ns_vals = torch.arange(0, self.Ns)

        # Add some lines to the axes
        for ex in range(examples):

            # Pick a random example from the dataset
            pe = int(torch.randint(0, data['L'].shape[0], size = (1,)))

            # Get the examples to plot in this round
            Ll = data['L'][pe, :].detach().numpy()
            Rl = data['R'][pe, :].detach().numpy()

            # Plot the coefficients side-by-side.
            axis_L1.bar(ns_vals,       Ll,         color=COLORS[ex], alpha=1.0, width=0.2)
            axis_L1.bar(ns_vals + 0.2, Lp[pe, :],  color=COLORS[ex], alpha=0.5, width=0.2)

            # Ll numpy to torch
            Ll_torch = torch.from_numpy(Ll)

            # Calculate difference between labels and predictions of coefficients
            delta_L = Ll_torch - Lp[pe, :]

            # Create color list, blue for positive, red for negative
            cc=['colors']*len(delta_L)
            for n,val in enumerate(delta_L):
                if val>0:
                    cc[n]='#b80000'
                elif val<=0:
                    cc[n]='#168AAD'

            # Plot the absolute difference between coefficients.
            axis_L2.bar(ns_vals, torch.abs((Ll_torch - Lp[pe, :])/Ll_torch), color=cc, alpha=0.7, width=0.2)

            # Plot the spectral function in the corresponding axes
            axis_R.plot(self.kernel.omega, Rl,        color=COLORS[ex], linestyle='-', alpha=1.0)
            axis_R.plot(self.kernel.omega, Rp[pe, :], color=COLORS[ex], linestyle='--', alpha=0.5)

            # Add two rectangles to the handles to show this examples
            handles.append(
                (
                    pat.Rectangle((0, 0), 2.0, 1.0, color=COLORS[ex], alpha=1.0),
                    pat.Rectangle((0, 0), 2.0, 1.0, color=COLORS[ex], alpha=0.5)
                )
            )

            # Add the label string to the sample
            labels.append(f'Label / Prediction: Example {ex}')

        fig.legend(
            handles, labels, numpoints=1, ncol=examples, frameon=False,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            bbox_to_anchor=(0, 0.95, 1, 0), loc='upper center'
        )

        return fig

    def __load_tensor(self, tensor_id: str, identifier: str, path: str) -> None:
        """ Load a tensor from the path into data. """

        # Create a path to the current tensor
        data_path = os.path.join(path, tensor_id + '_' + identifier + '.dat')

        # Assert the file exists
        assert os.path.exists(data_path)

        # Load the tensor into the correct memory address
        self.__data[tensor_id] = torch.load(data_path)

    def __generate_identifier(self, Nb: int, Ns: int, prefix: str, suffix: str):
        """ Generate a identifier name using some parameters. """
        # Add the prefix and the suffix to the name identifier
        prefix = f'{prefix}_' if prefix != '' else ''
        suffix = f'_{suffix}' if suffix != '' else ''

        # Generate the name identifier that should use the dataset
        name = f'b{Nb}_s{Ns}_w{self.Nw}_t{self.Nt}_mp{self.max_np}_fp{int(self.fixed_np)}'

        return f'{prefix}{name}{suffix}'

    def __check_correct_dataset(self, input_path: str):
        """ Check the dataset to be load is correct; it has the same attributes as the
        one we are now using.
        """
        # Assert that the dataset corresponds to the same parameters
        with open(os.path.join(input_path, 'info.json'), 'r') as f:
            # Load the content of the information file
            content = json.load(f)

            # Assert several conditions on the information
            assert content['Nt']       == self.Nt
            assert content['Nw']       == self.Nw
            assert content['limits']   == self.peak_limits
            assert content['max_np']   == self.max_np
            assert content['fixed_np'] == self.fixed_np
            assert content['kernel']   == str(self.kernel)
            assert all(pi == pc._peak_type for pi, pc in zip(content['peak_types'], self.peak_types))
            assert all(wi == wc for wi, wc in zip(content['w_range'], self.kernel.w_range))
    # -- }}}

    # -- Magic methods of the class {{{
    def __len__(self) -> int:
        """ The length of the object is the number of examples. """
        return self.__data['R'].shape[0] if self.is_generated else 0

    def __getitem__(self, idx: slice) -> tuple[torch.Tensor, torch.Tensor]:
        """ Accesing the object is a way of obtaining some examples from the correlation
        function and the spectral function. Indexing the dataset retrieves the same 
        examples from the correlation function and the coefficients

        --- Parameters:
        idx: slice
            Slice of the iterator to be retrieved.

        --- Returns:
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing the wanted examples for the correlation function and the 
            coefficients.
                                (C[idx,:,:], L[idx,:,:])
        """

        # To access the data, the data should be generated
        if not self.is_generated: raise ValueError("Dataset has not been generated.")

        # Get a view of the data in the correct dimensions
        C = self.__data['C'].view(self.Nb, 1, self.Nt)
        L = self.__data['L'].view(self.Nb, 1, self.Ns)

        return SpectralOut(C[idx,:,:], L[idx,:,:])

    def __str__(self) -> str:
        return f'<SpectralSet: {self.name}>' 

    def __repr__(self) -> str:
        return self.__str__()
    # -- }}}

    # -- Property methods of the class {{{
    @property
    def R(self) -> torch.Tensor:
        """ Dataset of spectral functions. """
        return self.__data['R']

    @property
    def C(self) -> torch.Tensor:
        """ Dataset of correlation functions. """
        return self.__data['C']

    @property
    def L(self) -> torch.Tensor:
        """ Dataset of coefficients in the expansion. """
        return self.__data['L']

    @property
    def U(self) -> torch.Tensor:
        """ Dataset of basis functions. """
        return self.__data['U']

    @property
    def is_generated(self) -> bool:
        """ Check whether the data has been generated or not. """
        return all(v is not None for v in self.__data.values())

    @property
    def name(self) -> str:
        """ Name identifier of the dataset. """
        return f'b{self.Nb}_s{self.Ns}_w{self.Nw}_t{self.Nt}_mp{self.max_np}_fp{int(self.fixed_np)}'

    @property
    def info(self) -> dict:
        return {
            'name':         self.name,
            'date':         str(datetime.date.today()),
            'is_generated': self.is_generated,
            'Nb':           self.Nb,
            'Nt':           self.Nt,
            'Nw':           self.Nw,
            'Ns':           self.Ns,
            'w_range':      self.kernel.w_range,
            'max_np':       self.max_np,
            'fixed_np':     self.fixed_np,
            'kernel':       str(self.kernel),
            'limits':       self.peak_limits,
            'peak_types':   [p._peak_type for p in self.peak_types],
        }

    @property
    def Nt(self) -> int:
        return self.kernel.Nt

    @property
    def Nw(self) -> int:
        return self.kernel.Nw

    @property
    def Ns(self) -> int:
        return self.__data['L'].shape[1] if self.is_generated else None

    @property
    def Nb(self) -> int:
        return self.__data['L'].shape[0] if self.is_generated else None
    # -- }}}

if __name__ == '__main__':
    pass
