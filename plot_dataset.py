import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from functools import reduce
from itertools import product
from torch.utils.data import Dataset


vox_views = [['sag', 'vox', 50], ['ax', 'vox', 50], ['cor', 'vox', 50]]


class PlotDataset:
    """Draw an interactive plot of a few subjects from a torchio dataset.
    Scrolling on the different images enable to navigate sections.

    Args:
        dataset: a torchio.ImagesDataset
        views: None or a sequence of views, each view is given as (view_type, coordinate_system, position),
            view_type is one among "sag", "ax" and "cor" which corresponds to sagittal, axial and coronal slices;
            coordinate_system is one among "vox" or "mm" and is responsible for placing the slice using position
            in term of voxel number or number of millimeters;
            position is either an integer between 0 and 100 if coordinate_system is "vox" or a float otherwise.
            If no value is provided, the default views are used:
            [['sag', 'vox', 50], ['ax', 'vox', 50], ['cor', 'vox', 50]].
        view_org: None or a sequence of length 2, responsible for the organisation of views in subplots.
            Default is (len(views), 1)
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
    """
    def __init__(self, dataset, views=None, view_org=None, image_key_name='t1',
                 subject_idx=5, subject_org=None, figsize=(16, 9), update_all_on_scroll=False):
        self.dataset = dataset
        self.views = views if views is not None else vox_views
        self.view_org = self.parse_view_org(view_org)
        self.image_key_name = image_key_name
        self.subject_idx = self.parse_subject_idx(subject_idx)
        self.subject_org = self.parse_subject_org(subject_org)
        self.figsize = figsize
        self.update_all_on_scroll = update_all_on_scroll

        self.imgs = {}
        self.figs_and_axes = []
        self.axes2view = {}
        self.cached_images_and_affines = {}

        self.coordinate_system_list = ["vox", "mm"]
        self.view_type_list = ["sag", "cor", "ax"]

        self.check_views()

        self.init_plot()
        self.scrolling = False

    def parse_view_org(self, view_org):
        if isinstance(view_org, Sequence) and len(view_org) == 2:
            return view_org
        else:
            return len(self.views), 1

    def parse_subject_idx(self, subject_idx):
        if isinstance(self.dataset, Dataset):
            data_len = len(self.dataset)
        else:
            data_len = len(self.dataset[self.image_key_name]['affine'])
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

    @staticmethod
    def view2slice(view_idx, idx, img):
        if view_idx == 0:
            view_slice = img[idx, :, :]
        elif view_idx == 1:
            view_slice = img[:, idx, :]
        else:
            view_slice = img[:, :, idx]
        return view_slice

    def check_views(self):
        for view_type, coordinate_system, _ in self.views:
            if coordinate_system not in self.coordinate_system_list:
                raise ValueError(
                    f'coordinate_system {coordinate_system} not recognized among {self.coordinate_system_list}'
                )
            if view_type not in self.view_type_list:
                raise ValueError(f'view_type {view_type} not recognized among {self.view_type_list}')

    @staticmethod
    def get_legend(subject, view_type, coordinate_system, position):
        complete_view_types = {'sag': 'sagittal', 'ax': 'axial', 'cor': 'coronal'}
        if coordinate_system == 'vox':
            return f'Subject {subject}: {complete_view_types[view_type]} section, {position}% voxels'
        else:
            return f'Subject {subject}: {complete_view_types[view_type]} section, {position} mm'

    def init_plot(self):
        """
        Initialize all the figures and axes following self.view_org and self.subject_org.
        Sets up the self.imgs dictionary and maps axes to views to access views from scroll events.
        Render the original views.
        """
        nb_subject_per_figure = np.product(self.subject_org)
        nb_figures = math.ceil(len(self.subject_idx) / np.product(self.subject_org))

        # Create figures and axes
        subplot_shape = (self.view_org[0] * self.subject_org[0], self.view_org[1] * self.subject_org[1])
        for i in range(nb_figures):
            fig, axes = plt.subplots(*subplot_shape, figsize=self.figsize)
            fig.tight_layout()
            axes = axes.reshape(subplot_shape)
            self.figs_and_axes.append([fig, axes])
            fig.canvas.mpl_connect('scroll_event', self.on_scroll)

            # Remove axis for all subplots
            for axis in axes.ravel():
                axis.axis('off')

        # Assign each view to its figure and axis
        for i, subject in enumerate(self.subject_idx):
            fig, axes = self.figs_and_axes[i // nb_subject_per_figure]
            i = i % nb_subject_per_figure

            for j, (view_type, coordinate_system, _) in enumerate(self.views):
                axis = axes[
                    j // self.view_org[1] + i // self.subject_org[1] * self.view_org[0],
                    j % self.view_org[1] + i % self.subject_org[1] * self.view_org[1]
                ]
                self.imgs[(subject, view_type, coordinate_system)] = {
                    'fig': fig,
                    'axis': axis,
                }
                self.axes2view[axis] = (subject, view_type, coordinate_system)

        # Render each view
        for subject, view in product(self.subject_idx, self.views):
            self.render_view(subject, view, init=True)

    def render_view(self, subject, view, init=False):
        """
        Generate the slice of interest from a view and render it.
        If self.update_all_on_scroll is True, all the views sharing the same view_type and
        coordinate_system are updated.

        :param subject: the subject whose view is rendered
        :param view: the view to render
        :param init: bool, if True draw the slices, otherwise update them
        """
        view_type, coordinate_system, position = view
        view_idx = self.view_type_list.index(view_type)
        img, affine = self.get_image_and_affine(subject)
        mapped_position, position = self.map_position(img, affine, position, view_idx, coordinate_system)
        view_slice = self.view2slice(view_idx, mapped_position, img)

        # Change slice orientation
        view_slice = np.flipud(view_slice.T)

        # Update image
        img_key = (subject, view_type, coordinate_system)
        self.imgs[img_key] = {
            **self.imgs[img_key],
            'position': position,
            'mapped_position': mapped_position,
            'view_idx': view_idx,
        }

        # Show or update image
        axis = self.imgs[img_key]['axis']
        text = self.get_legend(subject, view_type, coordinate_system, position)
        if init:
            axis.text(0.5, -0.1, text, size=10, ha="center", transform=axis.transAxes)
            self.imgs[img_key]['img'] = axis.imshow(view_slice, cmap='gray')
        else:
            if self.update_all_on_scroll:
                self.update_imgs(view_type, coordinate_system, position, mapped_position, view_idx)
            else:
                text_box = axis.texts[0]
                text_box.set_text(text)
                self.update_img(img_key, view_slice)

    def get_image_and_affine(self, subject):
        if subject not in self.cached_images_and_affines.keys():
            if isinstance(self.dataset, Dataset):
                obj = self.dataset[int(subject)][self.image_key_name]
                self.cached_images_and_affines[subject] = (obj['data'].numpy()[0], obj['affine'])
            else:
                obj = self.dataset[self.image_key_name]
                self.cached_images_and_affines[subject] = (obj['data'].numpy()[subject][0], obj['affine'][subject])
        return self.cached_images_and_affines[subject]

    @staticmethod
    def map_position(img, affine, position, view_idx, coordinate_system):
        if coordinate_system == "vox":
            if position < 0:
                position = 0
            if position >= 100:
                position = 99
            mapped_position = img.shape[view_idx] * position / 100
        else:
            position_vector = np.zeros(4)
            position_vector[view_idx] = position
            position_vector[3] = 1
            mapped_position = np.linalg.solve(affine, position_vector)[view_idx]

            # Clip mapped_position value to match image shape and compute clipped position
            if mapped_position < 0:
                mapped_position = 0
            if mapped_position >= img.shape[view_idx] - 0.5:
                mapped_position = img.shape[view_idx] - 1
            position_vector[view_idx] = mapped_position
            position = np.dot(affine, position_vector)[view_idx]

        return int(round(mapped_position)), int(round(position))

    def on_scroll(self, event):
        if not self.scrolling:
            self.scrolling = True
            # Get relevant axis
            subject, view_type, coordinate_system = self.axes2view[event.inaxes]
            position = self.imgs[(subject, view_type, coordinate_system)]['position']

            if event.button == 'down':
                view = (view_type, coordinate_system, position - 1)
            else:
                view = (view_type, coordinate_system, position + 1)

            self.render_view(subject, view)

    def update_img(self, img_idx, view_slice):
        img_dict = self.imgs[img_idx]
        img_dict['img'].set_data(view_slice)
        img_dict['fig'].canvas.draw()
        img_dict['fig'].canvas.flush_events()
        self.scrolling = False

    def update_imgs(self, view_type, coordinate_system, position, mapped_position, view_idx):
        img_keys = map(lambda s: (s, view_type, coordinate_system), self.subject_idx)
        fig_to_update = []
        for key in img_keys:
            img_dict = self.imgs[key]

            axis = img_dict['axis']
            text = self.get_legend(key[0], view_type, coordinate_system, position)
            text_box = axis.texts[0]
            text_box.set_text(text)

            img, _ = self.get_image_and_affine(key[0])
            view_slice = self.view2slice(view_idx, mapped_position, img)
            view_slice = np.flipud(view_slice.T)

            img_dict['img'].set_data(view_slice)
            fig = img_dict['fig']
            if fig not in fig_to_update:
                fig_to_update.append(fig)

        for fig in fig_to_update:
            fig.canvas.draw()
            fig.canvas.flush_events()

        self.scrolling = False
