import torch
import matplotlib.pyplot as plt
import matplotlib.patches as pat

from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines          import Line2D

# Colors used in the example plots
COLOR_1, COLOR_2, COLOR_3 = '#1E6091', '#40916C', '#BA181B'

def eliminate_mirror_axis(axis):
    ''' Eliminate mirror axis from an axis plot. '''

    # Eliminate the frame
    axis.set_frame_on(False)

    # Disable the drawing of ticks in the top of the plot
    axis.get_xaxis().tick_bottom()

    # Turn off the axis
    axis.axes.get_yaxis().set_visible(True)
    axis.axes.get_xaxis().set_visible(True)

    # Get the interval defining the axis
    xmin, xmax = axis.get_xaxis().get_view_interval()
    ymin, ymax = axis.get_yaxis().get_view_interval()

    # Add the lines for the axis
    axis.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    axis.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))

def plot_example(label: torch.Tensor, preds: torch.Tensor, U: torch.Tensor, omega: torch.Tensor, ne: int, ex: int = 0) -> plt.Figure:
    """ Plot an example of the validation set.  """

    # Generate the matplotlib figure
    fig = plt.figure(figsize=(16, 12))

    # Generate three (2 left 1 right) axis in the figure
    axis_L1 = fig.add_subplot(2, 2, 1)
    axis_L2 = fig.add_subplot(2, 2, 3)
    axis_R  = fig.add_subplot(1, 2, 2)

    # Set some properties in each of the axes
    axis_L1.set_xlabel(r'$n_s$')
    axis_L1.set_ylabel(r'$L(n_s)$')
    axis_L2.set_xlabel(r'$n_s$')
    axis_L2.set_ylabel(r'$|(\hat{L}(n_s) - L(n_s))|$')
    axis_R.set_xlabel(r'$\omega$')
    axis_R.set_ylabel(r'$\rho(\omega)$')
    axis_L1.grid('#fae1dd', alpha=0.3)
    axis_L2.grid('#fae1dd', alpha=0.3)
    axis_R.grid('#fae1dd',  alpha=0.3)

    # ns values to use in the plots
    ns_vals = torch.arange(0, label.shape[0])

    # Plot the coefficients side-by-side.
    axis_L1.bar(ns_vals,       label, color=COLOR_1, alpha=1.0, width=0.4)
    axis_L1.bar(ns_vals + 0.4, preds, color=COLOR_2, alpha=1.0, width=0.4)

    # Calculate difference between labels and predictions of coefficients
    delta_L = label - preds

    # Create color list, blue for positive, red for negative
    cc = [COLOR_1 if val > 0 else COLOR_3 for val in delta_L]

    # Plot the bars
    axis_L2.bar(ns_vals, delta_L.abs(), color=cc, alpha=1.0, width=0.5)

    # Plot the spectral function in the corresponding axes
    axis_R.plot(omega, label @ U, color=COLOR_1, linestyle='-',  alpha=1.0)
    axis_R.plot(omega, preds @ U, color=COLOR_2, linestyle='--', alpha=1.0)

    # Add two rectangles to the handles to show this examples
    handles = [
        (
            pat.Rectangle((0, 0), 2.0, 1.0, color=COLOR_1, alpha=1.0),
            pat.Rectangle((0, 0), 2.0, 1.0, color=COLOR_2, alpha=1.0)
        )
    ]

    # Add a legend to the figure
    fig.legend(
        handles, [f'{ne}: Label / Prediction'], numpoints=1, ncol=1, 
        frameon=False, handler_map={tuple: HandlerTuple(ndivide=None)},
        bbox_to_anchor=(0, 0.95, 1, 0), loc='upper center'
    )

    # Add a fake legend to the L2 to show the colours
    axis_L2.legend(
        [
            pat.Rectangle((0, 0), 2.0, 1.0, color=COLOR_1, alpha=1.0),
            pat.Rectangle((0, 0), 2.0, 1.0, color=COLOR_3, alpha=1.0)
        ],
        ['Positive', 'Negative'], ncol=1, loc='upper right', 
        frameon=False, fontsize='x-small'
    )

    # Remove all axis from the figures
    eliminate_mirror_axis(axis_L1)
    eliminate_mirror_axis(axis_L2)
    eliminate_mirror_axis(axis_R)

    # Return the figure to save it or manipulate it
    return fig

if __name__ == '__main__':
    pass
