import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider
from typing import Sequence
from functools import reduce, lru_cache
from torch.utils.data import DataLoader

vox_views = [['sag', 'vox', 50], ['ax', 'vox', 50], ['cor', 'vox', 50]]


class View:
    def __init__(self, view, type2idx, img, affine, subject, figure, axis, label=None, channel=0, show_text=False,
                 cmap='RdBu', alpha=0.3, threshold=0.01):
        self.view_type, self.coordinate_system, self.position = view
        self.view_idx = type2idx.index(self.view_type)
        self.img = img
        self.affine = affine
        self.subject = subject
        self.figure = figure
        self.axis = axis
        self.label = label
        self.channel = channel
        self.show_text = show_text

        self.init = True
        self.axis_img = None
        self.axis_label = None
        self.cmap = cmap
        self.alpha = alpha
        self.threshold = threshold

    def set_position(self, position):
        self.position = position

    def set_channel(self, channel):
        self.channel = channel

    @lru_cache(100)
    def mapped_position(self, position):
        if self.coordinate_system == "vox":
            if position < 0:
                position = 0
            if position >= 100:
                position = 99
            mapped_position = self.img.shape[self.view_idx] * position / 100
        else:
            position_vector = np.zeros(4)
            position_vector[self.view_idx] = position
            position_vector[3] = 1
            mapped_position = np.linalg.solve(self.affine, position_vector)[self.view_idx]

            # Clip mapped_position value to match image shape and compute clipped position
            if mapped_position < 0:
                mapped_position = 0
            if mapped_position >= self.img.shape[self.view_idx] - 0.5:
                mapped_position = self.img.shape[self.view_idx] - 1
            position_vector[self.view_idx] = mapped_position
            position = np.dot(self.affine, position_vector)[self.view_idx]

        return int(round(mapped_position)), int(round(position))

    @lru_cache(100)
    def img_slice(self, position):
        if self.view_idx == 0:
            view_slice = self.img[position, :, :]
        elif self.view_idx == 1:
            view_slice = self.img[:, position, :]
        else:
            view_slice = self.img[:, :, position]
        return np.flipud(view_slice.T)

    @property
    def legend(self):
        complete_view_types = {'sag': 'sagittal', 'ax': 'axial', 'cor': 'coronal'}
        if self.coordinate_system == 'vox':
            return f'Subject {self.subject}: {complete_view_types[self.view_type]} section, {self.position}% voxels'
        else:
            return f'Subject {self.subject}: {complete_view_types[self.view_type]} section, {self.position} mm'

    @lru_cache(100)
    def label_slice(self, position, channel):
        if self.view_idx == 0:
            view_slice = self.label[channel][position, :, :]
        elif self.view_idx == 1:
            view_slice = self.label[channel][:, position, :]
        else:
            view_slice = self.label[channel][:, :, position]
        return np.flipud(view_slice.T)

    def render(self):
        mapped_position, self.position = self.mapped_position(self.position)
        img_slice = self.img_slice(mapped_position)
        label_slice = None

        if self.label is not None:
            label_slice = self.label_slice(mapped_position, self.channel)

        if self.init:
            if self.show_text:
                self.axis.text(0.5, -0.1, self.legend, size=8, ha="center", transform=self.axis.transAxes)

            self.axis_img = self.axis.imshow(img_slice, cmap='gray')

            if self.label is not None:
                self.axis_label = self.axis.imshow(label_slice, cmap=self.cmap, alpha=self.alpha, clim=[self.threshold, 1])

            self.init = False

        else:
            if self.show_text:
                text_box = self.axis.texts[0]
                text_box.set_text(self.legend)
            self.axis_img.set_data(img_slice)

            if self.label is not None:
                self.axis_label.set_data(label_slice)

            self.axis.draw_artist(self.axis_img)

            if self.label is not None:
                self.axis.draw_artist(self.axis_label)

            self.figure.canvas.blit(self.axis.bbox)


class PlotDataset:
    """Draw an interactive plot of a few subjects from a torchio dataset.
    Scrolling on the different images enable to navigate sections.

    Args:
        dataset: a torchio.ImagesDataset or a DataLoader constructed from torchio.
        views: None or a sequence of views, each view is given as (view_type, coordinate_system, position),
            view_type is one among "sag", "ax" and "cor" which corresponds to sagittal, axial and coronal slices;
            coordinate_system is one among "vox" or "mm" and is responsible for placing the slice using position
            in term of voxel number or number of millimeters;
            position is either an integer between 0 and 100 if coordinate_system is "vox" or a float otherwise.
            If no value is provided, the default views are used:
            [['sag', 'vox', 50], ['ax', 'vox', 50], ['cor', 'vox', 50]].
        view_org: None or a sequence of length 2, responsible for the organisation of views in subplots.
            Default is (len(views), 1).
        image_key_name: a string that gives the key of the volume of interest in the dataset's samples.
        subject_idx: None, an integer or a sequence of integers. Defines which subjects are plotted.
            If subject_idx is a sequence of integers, it is directly used as the list of indexes,
            if subject_idx is an integer, subjects_idx subjects are taken at random in the dataset.
            Finally, if subject_idx is None, all subjects are taken.
            Default value is 5.
        subject_org: None or a sequence of length 2, responsible for the organisation of subjects in subplots.
            Default is (1, len(subject_idx))
        figsize: Sequence of length 2, size of the figure.
        update_all_on_scroll: bool, if True all views with the same view_type and coordinate_system are updated
            when scrolling on one of them. Doing so supposes that they all have the same shape. Default is False.
        add_text: Boolean to choose if you want the axis legend to be printed default True.
        label_key_name: a string that gives the key of the label of the volume of interest in the dataset's samples.
        alpha: overlay opacity, used when plotting label, default is 0.2.
        cmap: the colormap used to plot label, default is 'RdBu'.
        patch_sampler: a sampler used to generate patches from the volumes, if sampler is not None,
            the patches are superimposed on the volumes with white borders. Default is None.
        nb_patches: the number of patches to draw from each sample. Used only if patch_sampler is not None.
        threshold: the threshold below which label are not shown.
    """

    def __init__(self, dataset, views=None, view_org=None, image_key_name='image',
                 subject_idx=5, subject_org=None, figsize=(16, 9), update_all_on_scroll=False,
                 add_text=True, label_key_name=None, alpha=0.2, cmap='RdBu', patch_sampler=None, nb_patches=4,
                 threshold=0.01, datasample_list_size=0):
        self.dataset = dataset
        self.add_text = add_text
        self.views = views if views is not None else vox_views
        self.view_org = self.parse_view_org(view_org)
        self.image_key_name = image_key_name

        self.label_key_name = label_key_name

        if isinstance(self.label_key_name, str):
            self.label_key_name = [self.label_key_name]

        self.current_channel = 0

        self.cached_images_and_affines = {}
        self.cached_labels = {}

        self.threshold = threshold

        self.subject_idx = self.parse_subject_idx(subject_idx)

        if patch_sampler is not None:
            self.sample_patches(patch_sampler, nb_patches)

        self.datasample_list_size = datasample_list_size
        self.load_subjects()

        self.subject_org = self.parse_subject_org(subject_org)
        self.figsize = figsize
        self.update_all_on_scroll = update_all_on_scroll

        self.alpha = alpha
        self.cmap = cm.get_cmap(cmap)
        self.cmap.set_under(color='k', alpha=0)

        self.view_objects = {}
        self.figs_and_axes = []
        self.sliders = []
        self.axes2view_keys = {}

        self.coordinate_system_list = ["vox", "mm"]
        self.type2idx = ["sag", "cor", "ax"]

        self.updating_views = False
        self.sliding = False

        self.check_views()
        self.init_plot()

    def parse_view_org(self, view_org):
        if isinstance(view_org, Sequence) and len(view_org) == 2:
            return view_org
        else:
            return len(self.views), 1

    def parse_subject_idx(self, subject_idx):
        data_len = len(self.dataset)
        if hasattr(self.dataset, 'batch_size'):
            data_len *= self.dataset.batch_size
        if isinstance(subject_idx, Sequence):
            valid = reduce(lambda acc, val: acc and 0 <= val < data_len, subject_idx, True)
            if not valid:
                raise ValueError('Invalid index sequence')
            return subject_idx
        elif isinstance(subject_idx, int):
            return np.random.choice(range(data_len), min(data_len, subject_idx), replace=False).astype(int)
        else:
            return list(range(data_len))

    def parse_subject_org(self, subject_org):
        if isinstance(subject_org, Sequence) and len(subject_org) == 2:
            return subject_org
        else:
            return 1, len(self.subject_idx)

    def sample_patches(self, patch_sampler, nb_patches):
        for idx in self.subject_idx:
            subject = self.dataset[int(idx)]
            im = subject[self.image_key_name]['data'].numpy()[0].copy()
            labels = None

            if self.label_key_name is not None:
                labels = [subject[key]['data'].numpy()[0].copy() for key in self.label_key_name]

            for _ in range(nb_patches):
                patch = next(patch_sampler(subject))
                i_ini, j_ini, k_ini = patch['index_ini']
                i_fin, j_fin, k_fin = patch['index_ini'] + np.array(patch.spatial_shape)
                im_patch = patch[self.image_key_name]['data'].numpy()[0].copy()

                im_patch[:2, :, :] = 1
                im_patch[-2:, :, :] = 1
                im_patch[:, :2, :] = 1
                im_patch[:, -2:, :] = 1
                im_patch[:, :, :2] = 1
                im_patch[:, :, -2:] = 1

                im[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = im_patch

                if self.label_key_name is not None:
                    for label, key in zip(labels, self.label_key_name):
                        label_patch = patch[key]['data'].numpy()[0].copy()
                        label[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = label_patch

            self.cached_images_and_affines[idx] = (im, subject[self.image_key_name]['affine'].copy())

            if self.label_key_name is not None:
                self.cached_labels[idx] = labels

    def load_subjects(self):
        if isinstance(self.dataset, DataLoader):
            self.subject_idx = list(range(len(self.subject_idx)))
            max_idx = max(self.subject_idx)

            current_idx = 0
            for _, batch in enumerate(self.dataset):
                im = batch[self.image_key_name]['data']
                batch_len = len(im)
                for i in range(batch_len):
                    idx = current_idx + i
                    if idx > max_idx:
                        break
                    if idx not in self.cached_images_and_affines.keys():
                        self.cached_images_and_affines[idx] = im[i].numpy()[0].copy(), \
                            batch[self.image_key_name]['affine'][i].numpy().copy()
                        if self.label_key_name is not None:
                            self.cached_labels[idx] = batch[self.label_key_name]['data'][i].numpy()[0].copy()
                            self.cached_labels[idx] = [
                                batch[key]['data'][i].numpy()[0].copy() for key in self.label_key_name
                            ]
                current_idx += batch_len
                if current_idx > max_idx:
                    break
        elif self.datasample_list_size:  # should be set when using ListOf transform:
            new_idx = []
            for idx in self.subject_idx:
                for list_idx in range(self.datasample_list_size):
                    new_idx.append(idx*self.datasample_list_size + list_idx)
            self.subject_idx = new_idx
            for idx in self.subject_idx:
                if idx not in self.cached_images_and_affines.keys():
                    idx_subject = idx // self.datasample_list_size
                    subject = self.dataset[idx_subject]
                    print('loading suj {} with {} transform'.format(subject[0][self.image_key_name]['path'],
                                                                    len(subject)))

                    for idx_list in range(len(subject)):
                        suj = subject[idx_list]
                        self.cached_images_and_affines[idx+idx_list] = suj[self.image_key_name]['data'].numpy()[0].copy(), \
                                                              suj[self.image_key_name]['affine'].copy()
                        if self.label_key_name is not None:
                            self.cached_labels[idx+idx_list] = [suj[key]['data'].numpy()[0].copy() for key in self.label_key_name]

        else:
            for idx in self.subject_idx:
                if idx not in self.cached_images_and_affines.keys():
                    subject = self.dataset[int(idx)]
                    self.cached_images_and_affines[idx] = subject[self.image_key_name]['data'].numpy()[0].copy(), \
                        subject[self.image_key_name]['affine'].copy()
                    if self.label_key_name is not None:
                        self.cached_labels[idx] = [subject[key]['data'].numpy()[0].copy() for key in self.label_key_name]

    def check_views(self):
        for view_type, coordinate_system, _ in self.views:
            if coordinate_system not in self.coordinate_system_list:
                raise ValueError(
                    f'coordinate_system {coordinate_system} not recognized among {self.coordinate_system_list}'
                )
            if view_type not in self.type2idx:
                raise ValueError(f'view_type {view_type} not recognized among {self.type2idx}')

    def init_plot(self):
        """
        Initialize all the figures and axes following self.view_org and self.subject_org.
        Sets up the self.view_objects dictionary and maps axes to views to access views from scroll events.
        Render the original views.
        """
        nb_subject_per_figure = np.product(self.subject_org)
        nb_figures = math.ceil(len(self.subject_idx) / nb_subject_per_figure)

        # Create figures and axes
        subplot_shape = (self.view_org[0] * self.subject_org[0], self.view_org[1] * self.subject_org[1])
        for i in range(nb_figures):
            fig, axes = plt.subplots(*subplot_shape, figsize=self.figsize)
            fig.tight_layout()
            axes = axes.reshape(subplot_shape)
            self.figs_and_axes.append([fig, axes])

            # Add event handlers
            fig.canvas.mpl_connect('scroll_event', self.on_scroll)
            fig.canvas.mpl_connect('key_press_event', self.on_key_press)

            # Add sliders to set overlay display threshold
            if self.label_key_name is not None:
                axis = plt.axes([0.03, 0.05, 0.02, 0.9])
                slider = Slider(axis, 'Threshold', 0, 1, valinit=self.threshold, valstep=0.01, orientation='vertical')
                slider.on_changed(self.on_slide)
                self.sliders.append(slider)

            # Remove axis for all subplots
            for axis in axes.ravel():
                axis.axis('off')

            # Remove space between subplots
            if self.label_key_name is None:
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            else:
                plt.subplots_adjust(left=0.04, right=0.96, bottom=0, top=1, wspace=0, hspace=0)

        # Assign each view to its figure and axis
        for i, subject in enumerate(self.subject_idx):
            fig, axes = self.figs_and_axes[i // nb_subject_per_figure]
            i = i % nb_subject_per_figure

            for j, view in enumerate(self.views):
                axis = axes[
                    j // self.view_org[1] + i // self.subject_org[1] * self.view_org[0],
                    j % self.view_org[1] + i % self.subject_org[1] * self.view_org[1]
                ]
                view_type, coordinate_system, _ = view
                view_key = (subject, view_type, coordinate_system)
                img, affine = self.cached_images_and_affines[subject]
                label = None

                if self.label_key_name is not None:
                    label = self.cached_labels[subject]

                self.view_objects[view_key] = View(view, self.type2idx, img, affine, subject, fig, axis, label=label,
                                       channel=self.current_channel, show_text=self.add_text,
                                       cmap=self.cmap, alpha=self.alpha, threshold=self.threshold)
                self.axes2view_keys[axis] = view_key

        # Render each view
        for view_object in self.view_objects.values():
            view_object.render()

        # Display colorbar
        if self.label_key_name is not None:
            # Create a fake label to make the colorbar invariant to threshold changes
            axis = plt.axes([0., 0., 0., 0.])
            fake_label = axis.imshow(np.linspace(0, 1, 100).reshape(10, 10), cmap=self.cmap, alpha=self.alpha)
            fake_label.set_visible(False)
            for fig, axes in self.figs_and_axes:
                color_bar_axis = fig.add_axes([0.95, 0.05, 0.02, 0.9])
                fig.colorbar(fake_label, cax=color_bar_axis)

        # Draw canvas
        for fig, axes in self.figs_and_axes:
            fig.canvas.draw()

    def update(self, view_key, update_function):
        if self.update_all_on_scroll:
            subject, view_type, coordinate_system = view_key
            view_keys = [(s, view_type, coordinate_system) for s in self.subject_idx]
        else:
            view_keys = [view_key]

        fig_to_update = []

        for key in view_keys:
            view_object = self.view_objects[key]
            update_function(view_object)
            view_object.render()

            fig = view_object.figure
            if fig not in fig_to_update:
                fig_to_update.append(fig)

        for fig in fig_to_update:
            fig.canvas.flush_events()

        self.updating_views = False

    def on_scroll(self, event):
        if not self.updating_views:
            view_key = self.axes2view_keys.get(event.inaxes)
            if view_key is not None:
                self.updating_views = True

                if event.button == 'down':
                    delta_position = -1
                else:
                    delta_position = 1

                self.update(
                    view_key,
                    lambda x: x.set_position(x.position + delta_position)
                )

    def on_key_press(self, event):
        if self.label_key_name is not None and not self.updating_views:
            if event.key == 'down' or event.key == 'up':
                view_key = self.axes2view_keys.get(event.inaxes)
                if view_key is not None:
                    self.updating_views = True
                    if event.key == 'down':
                        delta_channel = -1
                    else:
                        delta_channel = 1

                    self.update(
                        view_key,
                        lambda x: x.set_channel((x.channel + delta_channel) % len(self.label_key_name))
                    )

    def on_slide(self, val):
        if not self.sliding:
            self.sliding = True
            for slider in self.sliders:
                slider.set_val(val)
            self.threshold = val
            for key in self.view_objects.keys():
                self.view_objects[key].axis_label.set_clim([self.threshold, 1])
            self.sliding = False
