from collections import defaultdict, Counter
from typing import Dict

from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import FuncFormatter
from plot.inputdescriptor import InputDescriptor
from plot.operator import DictMapper
from matplotlib import rcParams


def create_scatter(plt, data, descriptors: Dict[str, InputDescriptor], sort=None, horizontal=False):
    args = defaultdict(list)
    for row in data:
        for key, descriptor in descriptors.items():
            args[key].append(descriptor.getter(row))

    if sort is not None:
        sort(args)

    scatter_list = []
    fig, ax = plt.subplots(figsize=(8, len(Counter(args['y']).keys()) * .5)if horizontal else (len(Counter(args['x']).keys()) * .75, 8))
    if 'markers' in args:
        index_list = defaultdict(list)
        for i, marker in enumerate(args['markers']):
            index_list[marker].append(i)

        del args['markers']
        for marker, index in index_list.items():
            sub_args = {key: [values[i] for i in index] for key, values in args.items()}
            scatter = ax.scatter(**sub_args, marker=marker, label=descriptors['markers'].mapper.reverse(marker))
            scatter_list.append(scatter)
    else:
        scatter = ax.scatter(**args)
        scatter_list.append(scatter)

    previous_paths = scatter.get_paths()
    scatter.set_paths([MarkerStyle('s').get_path()])
    if 'c' in descriptors:
        create_scatter_legend(ax, scatter, 'colors', descriptors['c'], "lower left")
    if 's' in descriptors and ('c' not in descriptors or descriptors['s'].label != descriptors['c'].label):
        create_scatter_legend(ax, scatter, 'sizes', descriptors['s'], "upper left")
    scatter.set_paths(previous_paths)

    plt.xlabel(descriptors['x'].label)
    plt.ylabel(descriptors['y'].label)

    # if markers are used, the color applied on the legend of the markers are somehow random and dont fit well
    # try to fix this by applying specific color to the scatter components -> after creation revert it
    prev_facecolor_lists = [s.get_facecolor() for s in scatter_list]
    for s in scatter_list: s.set_facecolor('black')
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    for i, s in enumerate(scatter_list): s.set_facecolor(prev_facecolor_lists[i])

    return fig


def create_scatter_legend(ax, scatter, prop, descriptor: InputDescriptor, loc):
    if prop == 'colors' and hasattr(descriptor.mapper, 'mapping') and isinstance(descriptor.mapper, DictMapper):
        mapping = descriptor.mapper.mapping
        handels = [Line2D([0], [0], ls="", color=color, ms=rcParams["lines.markersize"], marker=scatter.get_paths()[0]) for color in mapping.values()]
        args = (handels, mapping.keys())
    else:
        args = scatter.legend_elements(
            prop=prop
            , fmt=FuncFormatter(descriptor.mapper.reverse)
            , num=None
        )

    ax.add_artist(ax.legend(*args, loc=loc, title=descriptor.label, bbox_to_anchor=(1, .65)))
