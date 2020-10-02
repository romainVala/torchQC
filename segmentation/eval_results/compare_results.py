import pandas as pd
import collections
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import json

sns.set()

COLORS = ['#ffb380', '#6666ff', '#94b8b8', '#bf80ff', '#ffff80', '#d9b38c']


def _get_file(results_dir):
    csv_pattern = 'Whole_image_*.csv'
    files = glob.glob(os.path.join(results_dir, csv_pattern))
    if len(files) > 0:
        return files[-1]
    else:
        csv_pattern = 'Val_*.csv'
        return glob.glob(os.path.join(results_dir, csv_pattern))[-1]


def _get_order_from_col(df, col, to_round=False):
    if to_round:
        order = df[col].round().sort_values().astype(int).unique().astype(str)
        df[col] = df[col].round().astype(int).astype(str)
    else:
        order = df[col].sort_values().unique().astype(str)
        df[col] = df[col].astype(str)
    return order


def _draw_catplot(df, col, metric, order=None, ylim=None, hue='model',
                  hue_order=None, palette=None):
    fig = sns.catplot(
        x=col, y=metric, hue=hue, kind='bar', col='mode', data=df,
        order=order, hue_order=hue_order, palette=palette
    )
    if ylim is not None:
        plt.ylim(ylim)
    return fig


def _flatten(d, parent_key='', sep='_'):
    """ From https://stackoverflow.com/a/6027615"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _adjust_lightness(color, amount=0.5):
    """ Adjust color lightness, taken from
    https://stackoverflow.com/a/49601444"""
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def _get_color_palette(n, group_size=1):
    if n % group_size != 0:
        raise ValueError(f'n must be a multiple of group_size but {n}'
                         f'and {group_size} were found.')
    if n // group_size > len(COLORS):
        raise ValueError(f'{len(COLORS)} colors are available but '
                         f'{n // group_size} colors are needed.')
    colors = []
    if group_size == 1:
        light_range = [1]
    else:
        light_range = np.linspace(0.9, 1.1, group_size)
    for i in range(n // group_size):
        for lightness in light_range:
            colors.append(_adjust_lightness(COLORS[i], lightness))
    return colors


def _parse_metric(df, metric, label='GM', spacing=(1, 1, 1)):
    if metric == 'dice':
        return 1 - df[f'metric_dice_loss_{label}']
    elif metric == 'bin_dice':
        return 1 - df[f'metric_bin_dice_loss_{label}']
    elif metric == 'volume_difference':
        return df[f'occupied_volume_{label}'] \
               - df[f'predicted_occupied_volume_{label}']
    elif metric == 'abs_volume_difference':
        return np.abs(
            df[f'occupied_volume_{label}'] -
            df[f'predicted_occupied_volume_{label}']
        )
    elif metric == 'volume_difference_in_mm3':
        return (df[f'occupied_volume_{label}']
                - df[f'predicted_occupied_volume_{label}']) \
               * np.prod(df['shape'].str.strip('[]')
                         .str.split(' ', expand=True).astype(int).T) \
               * np.prod(spacing)
    elif metric == 'abs_volume_difference_in_mm3':
        return np.abs(df[f'occupied_volume_{label}']
                      - df[f'predicted_occupied_volume_{label}']) \
               * np.prod(df['shape'].str.strip('[]')
                         .str.split(' ', expand=True).astype(int).T) \
               * np.prod(spacing)
    elif metric == 'label_volume_in_mm3':
        return np.abs(df[f'occupied_volume_{label}']) \
               * np.prod(df['shape'].str.strip('[]')
                         .str.split(' ', expand=True).astype(int).T) \
               * np.prod(spacing) / 100000
    elif metric == 'predicted_volume_in_mm3':
        return np.abs(df[f'predicted_occupied_volume_{label}']) \
               * np.prod(df['shape'].str.strip('[]')
                         .str.split(' ', expand=True).astype(int).T) \
               * np.prod(spacing) / 100000
    elif metric == 'volume_ratio':
        return df[f'predicted_occupied_volume_{label}'] \
            / df[f'occupied_volume_{label}']
    else:
        return df[f'metric_{metric}_{label}'] if len(label)>0 else df[f'metric_{metric}']


def compare_results(results_dirs, filename, metrics, names=None):
    """ Create a csv file out of a list of result files and metrics.

    Example:
        >>> from segmentation.eval_results.occupation_stats import occupation_stats, \
        >>>     abs_occupation_stats, dice_score_stats, bin_dice_score_stats
        >>> from segmentation.eval_results.compare_results import compare_results
        >>> from itertools import product
        >>> model_prefixes = ['bin_synth', 'bin_dice_pve_synth', 'pve_synth']
        >>> GM_levels = ['03', '06', '09']
        >>> noise_levels = ['01', '05', '1']
        >>> modes = ['t1', 't2']
        >>> results_dirs, names = [], []
        >>> root = '/home/fabien.girka/data/segmentation_tasks/RES_1.4mm/'
        >>> filename = f'{root}compare_models.csv'
        >>> metrics = [dice_score_stats, bin_dice_score_stats, occupation_stats, abs_occupation_stats]
        >>> for mode, GM_level, noise, prefix in product(modes, GM_levels, noise_levels, model_prefixes):
        >>>     results_dirs.append(f'{root}{prefix}_data_64_common_noise_no_gamma/eval_on_{mode}_like_data_GM_{GM_level}_{noise}_noise')
        >>>     names.append(f'{prefix}_{mode}_GM_{GM_level}_{noise}_noise')
        >>> compare_results(results_dirs, filename, metrics, names)
    """
    results = {}
    for metric in metrics:
        for i, results_dir in enumerate(results_dirs):
            if names is not None:
                name = names[i]
            else:
                name = Path(results_dir).parent.name
            results[f'{name}_{metric.__name__}'] = _flatten(metric(results_dir))

    df = pd.DataFrame.from_dict(results)
    df.to_csv(filename)


def plot_dice_against_patch_overlap():
    files = glob.glob(
        '/home/fabien.girka/data/segmentation_tasks/RES_1.4mm/'
        'eval_patch_overlap/po_*/Whole*.csv')
    files.sort(key=os.path.getmtime)
    dice_scores = []
    overlaps = [4, 8, 12, 16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                42, 44, 46, 48]
    for file in files:
        df = pd.read_csv(file, index_col=0)
        dice_scores.append(1 - df['metric_dice_loss_GM'])
    plt.figure()
    plt.plot(overlaps, dice_scores)


def aggregate_csv_files(pattern, filename, fragment_position=-3):
    """Create a CSV file out of smaller ones and add columns 'model',
    'GM', 'SNR' and 'mode'. Small CSV files are used using a glob pattern
    or a list of glob patterns.

    Example:
        >>> from segmentation.eval_results.compare_results import aggregate_csv_files
        >>> pattern = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/RES_1mm/*/*/eval.csv'
        >>> aggregate_csv_files(pattern, 'some_path.csv')
    """
    if isinstance(pattern, str):
        files = glob.glob(pattern)
    else:
        files = [f for p in pattern for f in glob.glob(p)]
    data_frames = [pd.read_csv(file, index_col=0) for file in files]
    for i, file in enumerate(files):
        name = file.split('/')[fragment_position]

        # Get information
        fragments = name.split('_')

        if 'T_RandomLabelsToImage' in data_frames[i]:
            ddic = json.loads(data_frames[i]['T_RandomLabelsToImage'].values[0])
            gm = ddic['random_parameters_images_dict']['mean'][0]
            wm = ddic['random_parameters_images_dict']['mean'][2]
            GM_level = float('{:.1f}'.format(gm))
            mode = 'T1' if wm > gm else 'T2'
        else:
            GM_level = fragments[1]
            GM_level = float(f'0.{GM_level[-1]}')

            mode = fragments[2]
        if 'T_RandomNoise' in data_frames[i]:
            ddic = json.loads(data_frames[i]['T_RandomNoise'].values[0])
            std = ddic['std']
            SNR_level = float('{:.3f}'.format(std))
        else:
            SNR_level = fragments[4]

        model_name = '_'.join(fragments[6:])

        # Parse information
        #SNR_level = int(SNR_level)

        data_frames[i]['model'] = model_name
        data_frames[i]['GM'] = GM_level
        data_frames[i]['SNR'] = SNR_level
        data_frames[i]['mode'] = mode

    final_data_frame = pd.concat(data_frames)
    final_data_frame.to_csv(filename)


def create_data_frame(results_dirs, metric, ref_tissue='WM', spacing=(1, 1, 1), label='GM'):
    files = [_get_file(r) for r in results_dirs]
    data_frames = [pd.read_csv(file, index_col=0) for file in files]
    final_data_frames = []
    for i, file in enumerate(files):
        name = Path(file).parent.name

        # Get information
        fragments = name.split('_')

        noise_level = fragments[5]
        GM_level = fragments[3]
        mode = fragments[1]
        model_name = '_'.join(fragments[6:])

        # Parse information
        noise_level = float(f'0.{noise_level}')
        GM_level = float(f'0.{GM_level}')
        if ref_tissue == 'WM':
            if mode == 't1':
                ref_tissue_level = 1.
            else:
                ref_tissue_level = 0.2
        elif ref_tissue == 'CSF':
            if mode == 't1':
                ref_tissue_level = 0.2
            else:
                ref_tissue_level = 1.
        else:
            raise ValueError(
                'Only "WM" and "CSF" are acceptable ref_tissue values')

        # Compute SNR and CNR
        SNR = GM_level / noise_level
        CNR = abs(GM_level - ref_tissue_level) / noise_level

        # Get values
        values = _parse_metric(data_frames[i], metric, label, spacing)

        final_data_frames.append(
            pd.DataFrame({
                metric: values,
                'model': model_name,
                'noise': noise_level,
                'GM': GM_level,
                'SNR': SNR,
                'CNR': CNR,
                'mode': mode
            })
        )

    return pd.concat(final_data_frames)


def plot_value_vs_CNR(results_dirs, metric, ref_tissue='WM', ylim=None,
                      save_fig=None):
    """ Draw a bar plot of values from a given metric against the CNR level
    between GM and a tissue of reference (WM or CSF).

    Example:
        >>> from segmentation.eval_results.compare_results import plot_value_vs_CNR
        >>> import glob
        >>> results_dirs = glob.glob('/home/fabien.girka/data/segmentation_tasks/RES_1mm/eval_models/data_t*')
        >>> plot_value_vs_CNR(results_dirs, 'dice_loss', ylim=(0, 0.1),
        >>>     save_fig='/home/fabien.girka/data/segmentation_tasks/RES_1mm/eval_models/dice_against_CNR')
    """
    df = create_data_frame(results_dirs, metric, ref_tissue)
    order = _get_order_from_col(df, 'CNR', to_round=True)
    palette = _get_color_palette(len(df['model'].unique()))
    fig = _draw_catplot(df, 'CNR', metric, order, ylim, palette=palette)
    if save_fig is not None:
        fig.savefig(save_fig)
    plt.show()


def plot_value_vs_SNR(results_dirs, metric, ylim=None, save_fig=None):
    """ Draw a bar plot of values from a given metric against the GM SNR level.

    Example:
        >>> from segmentation.eval_results.compare_results import plot_value_vs_SNR
        >>> import glob
        >>> results_dirs = glob.glob('/home/fabien.girka/data/segmentation_tasks/RES_1mm/eval_models/data_t*')
        >>> plot_value_vs_SNR(results_dirs, 'dice_loss', ylim=(0, 0.1),
        >>>     save_fig='/home/fabien.girka/data/segmentation_tasks/RES_1mm/eval_models/dice_against_SNR')
    """
    df = create_data_frame(results_dirs, metric)
    order = _get_order_from_col(df, 'SNR', to_round=True)
    palette = _get_color_palette(len(df['model'].unique()))
    fig = _draw_catplot(df, 'SNR', metric, order, ylim, palette=palette)
    if save_fig is not None:
        fig.savefig(save_fig)
    plt.show()


def plot_value_vs_GM_level(results_dirs, metric, ylim=None, save_fig=None, label='GM'):
    """ Draw a bar plot of values from a given metric against the GM level.

    Example:
        >>> from segmentation.eval_results.compare_results import plot_value_vs_GM_level
        >>> import glob
        >>> results_dirs = glob.glob('/home/fabien.girka/data/segmentation_tasks/RES_1mm/eval_models/data_t*')
        >>> results_dirs += glob.glob('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_samseg14mm/evalON_*')
        >>> plot_value_vs_GM_level(results_dirs, 'dice_loss', ylim=(0, 0.1),
        >>>     save_fig='/home/fabien.girka/data/segmentation_tasks/RES_1mm/eval_models/dice_against_GM')
    """
    df = create_data_frame(results_dirs, metric, label=label)
    df['model_and_noise'] = df['model'].str.cat(
        df['noise'].astype(str), sep='_')
    order = _get_order_from_col(df, 'GM', to_round=False)
    hue_order = _get_order_from_col(df, 'model_and_noise', to_round=False)
    palette = _get_color_palette(
        len(df['model_and_noise'].unique()), len(df['noise'].unique()))
    fig = _draw_catplot(df, 'GM', metric, order, ylim, 'model_and_noise',
                        hue_order, palette)
    if save_fig is not None:
        fig.savefig(save_fig)
    plt.show()


def plot_metric_against_GM_level(result_file, metrics, ylim=None, save_fig=None):
    """ Draw a bar plot of values from a given metric against the GM level
    from a CSV result file.

    Example:
        >>> from segmentation.eval_results.compare_results import plot_metric_against_GM_level
        >>> result_file = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/res1mm_all.csv'
        >>> plot_metric_against_GM_level(result_file, 'metric_dice_loss_GM', ylim=(0, 0.1))
    """
    df = pd.read_csv(result_file, index_col=0)
    df.sort_values(['model', 'SNR'], axis=0, inplace=True)
    df['model_and_SNR'] = df['model'].str.cat(df['SNR'].astype(str), sep='_')
    palette = _get_color_palette(
        len(df['model_and_SNR'].unique()), len(df['SNR'].unique()))

    for metric in metrics:
        if metric not in df:
            print(f'mission column {metric}')
            continue
        fig = _draw_catplot(df, 'GM', metric, ylim=ylim, hue='model_and_SNR',
                            palette=palette)
        if save_fig is not None:
            fig.savefig(save_fig + '_' + metric + '.png')
        plt.show()
