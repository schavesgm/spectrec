# Import built-in modules
import re
import time
import datetime
from collections import deque, defaultdict
from typing import Union, Iterable, Generator

# Import third-party modules
import torch
import torch.distributed as dist

class TimeSeries:
    """ Time series object to extract some information from it. """

    def __init__(self, num_lags: int = 20, out_fmt: str = None):

        # Save the format to be used in the printing
        self.fmt        = out_fmt if out_fmt is not None else '{median:.4f} ({global_average:.4f})'
        self.num_lags   = num_lags
        self.saved_lags = deque(maxlen=num_lags)

        # Save all values and the total counts in the time series
        self.sum_values, self.total_count = 0.0, 0

    def update(self, value: Union[int, float, torch.Tensor], copies: int = 1):
        """ Add the value to the time series. """

        # If the value is a torch Tensor, then get the item
        value = value.item() if isinstance(value, torch.Tensor) else value

        # Add all copies to the 
        [self.saved_lags.append(value) for _ in range(copies)]

        # Add the value to the total sum and total count
        self.sum_values  += value * copies
        self.total_count += copies

    def synchronise_between_processes(self):
        """ Synchronise all time series shared among different processes. The queue is not sync. """
        if not dist.is_initialized() or not dist.is_available():
            return

        # Get the total count and sum of values for all tensors
        count_sum = torch.tensor([self.total_count, self.sum_values], dtype=torch.float64, device='cuda')

        # Put a barrier to synchronise all processes
        dist.barrier()

        # Each process gets the sum of total_count and sum_values
        dist.all_reduce(count_sum)

        # Trasform the tensor to a list and set the attributes
        count_sum = count_sum.tolist()

        # Set the synchronised count and values
        self.total_count, self.sum_values = count_sum[0], count_sum[1]

    @property
    def median(self) -> torch.Tensor:
        """ Compute the median over the lags inside the queue. """
        return torch.tensor(list(self.saved_lags), dtype=torch.float32).median().item()


    @property
    def average(self) -> torch.Tensor:
        """ Compute the average over the lags inside the queue. """
        return torch.tensor(list(self.saved_lags), dtype=torch.float32).mean().item()

    @property
    def global_average(self) -> torch.Tensor:
        """ Compute the average over all points in the time series. """
        return torch.tensor(self.sum_values / self.total_count).item()

    @property
    def max(self) -> float:
        """ Get the maximum value in the queue. """
        return max(self.saved_lags)

    @property
    def min(self) -> float:
        """ Get the minimum value in the queue. """
        return min(self.saved_lags)

    def __str__(self) -> str:
        """ Return the formated string using all statistics in the class. """
        return self.fmt.format(
            median=self.median,
            average=self.average,
            global_average=self.global_average,
            total_count=self.total_count,
            max=self.max,
            min=self.min
        )

class TrainLogger:

    def __init__(self, delimiter: str = '    ', log_every: int = 10):

        # Save the delimiter and the pace of log
        self.delimiter = delimiter
        self.log_every = log_every

        # Time series that will contain the loss function
        self.loss_ts = TimeSeries()

    def update_loss(self, loss: Union[int, float, torch.Tensor]):
        """ Add the value to the loss time series. """
        self.loss_ts.update(loss)

    def synchronise_between_processes(self):
        self.loss_ts.synchronise_between_processes()

    def log_information(self, iterable: Iterable, header: str = '') -> Generator:
        """ Create a generator around an iterable that will output some data
        every some iterations.
        """

        # Counter used in the iteration and conversion to Megabytes
        iteration    = 0
        
        # Get the time of the iteration
        start_time = time.time()

        # Generate a time series
        iter_time = TimeSeries(out_fmt='{average:.4f}')

        # Generate the log format
        log_fmt = self.delimiter.join(
            [
                '',
                header,
                '[{0:' + str(len(str(len(iterable)))) + 'd}/{1}]',
                'metrics: {metrics}',
                'iter_time: {time}'
            ]
        )

        # Iterate for each object in the iterable
        for obj in iterable:

            # The iteration starts right now
            start_iter = time.time()

            # Yield the object, manipulate it and resume after this
            yield obj

            # Update the iteration time
            iter_time.update(time.time() - start_iter)
            
            # Log information every log_every iteration and in the last iteration
            if iteration % self.log_every == 0 or iteration == len(iterable) - 1:

                # Format the message to be logged
                out_msg = log_fmt.format(
                    iteration, len(iterable),
                    metrics=str(self.loss_ts), time=str(iter_time)
                )

                if torch.cuda.is_available():
                    out_msg = out_msg + self.delimiter + 'mem: {0:.0f}'.format(
                        torch.cuda.max_memory_allocated() / (1024 ** 2)
                    )

                # Print the output message
                print(out_msg, flush=True)

            # Go to the next iteration
            iteration += 1

        # Get the total time elapsed in the whole iteration over Iterable
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        # Print some final information to the console
        print(
            '  --  END PROCESSING BATCH: Loss: {}    Total_time: {} - {:.4f} s/iter'.format(
                str(self.loss_ts), total_time_str, total_time / len(iterable)
            ), flush=True
        )

if __name__ == '__main__':
    pass
