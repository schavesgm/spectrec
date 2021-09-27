# Import some built-in modules
import os

# Import some third-party modules
import torch.nn as nn
import torch


class Network(nn.Module):
    """ Base class used to define a network. """

    # Define the network name
    _net_id: str = 'baseNet'

    def set_params(self, state_dict: dict, eliminate_str: str = 'module.') -> None:
        """ Set the parameters of the model using the state_dict provided. """

        # Generate a new dictionary to clean the keys
        clean_state_dict: dict[str, torch.Tensor] = {}

        # Iterate to clean the data
        for key, value in state_dict.items():
            clean_state_dict[key.replace(eliminate_str, '', 1)] = value
        
        # Load the state dictionary
        self.load_state_dict(clean_state_dict)

    def save_params(self, identifier: str, path: str = './status/network'):
        """ Set the current status of the network into a given file. """

        # Path where the data will be stored
        save_path = os.path.join(path, self._net_id)

        # Check if the path exists
        if not os.path.exists(save_path): os.makedirs(save_path)

        # Save the weights into the path
        torch.save(self.state_dict(), os.path.join(save_path, f'{identifier}.pt'))

    def load_params(self, identifier: str, path: str = './status/network', verbose: bool = True):
        """ Load the parameters of a network with given identifier. """

        # Path to the given file
        load_path = os.path.join(path, self._net_id, f'{identifier}.pt')
        print(load_path)

        # Try loading the parameters
        try:
            
            # Load the parameters
            self.load_state_dict(torch.load(load_path))

            # Evaluate the parameters in the model
            self.eval()

            if verbose: 
                print(f'Loaded {self._net_id}:{identifier} instance', flush=True)
        
        # If the file is not found, just don't do anything
        except FileNotFoundError:
            if verbose:
                print(f'Network {self._net_id}:{identifier} not found. Not instance', flush=True)

    @property
    def num_weights(self) -> int:
        """ Show the total number of learnable parameters. """
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    pass
