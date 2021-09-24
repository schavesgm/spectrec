def eliminate_mirror_axis(axis):
    ''' Eliminate mirror axis from an axis plot. '''

    from matplotlib.lines import Line2D

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

if __name__ == '__main__':
    pass
