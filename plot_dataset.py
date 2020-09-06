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
    def __init__(self, view, type2idx, subject, figure, axis, channel,
                 show_text, cmap, alpha, threshold):
        self.view_type, self.coordinate_system, self.position = view
        self.view_idx = type2idx.index(self.view_type)
        self.subject = subject
        self.figure = figure
        self.axis = axis
        self.channel = channel
        self.show_text = show_text
        self.is_drawn = False

        self.bg_cache = self.figure.canvas.copy_from_bbox(self.axis.bbox)

        self.axis_img = None
        self.axis_label = None
        self.cmap = cmap
        self.alpha = alpha
        self.threshold = threshold

        self.img = None
        self.affine = None
        self.label = None

    def set_position(self, position):
        self.position = position

    def set_channel(self, channel):
        self.channel = channel

    def set_img_affine_and_label(self, img, affine, label):
        self.img = img
        self.affine = affine
        self.label = label

    def set_is_drawn(self, is_drawn):
        self.is_drawn = is_drawn

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
            mapped_position = np.linalg.solve(
                self.affine, position_vector)[self.view_idx]

            # Clip mapped_position to match image shape and clip position
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
        complete_view_types = {
            'sag': 'sagittal', 'ax': 'axial', 'cor': 'coronal'}
        if self.coordinate_system == 'vox':
            return f'Subject {self.subject}: ' \
                   f'{complete_view_types[self.view_type]} section, ' \
                   f'{self.position}% voxels'
        else:
            return f'Subject {self.subject}: ' \
                   f'{complete_view_types[self.view_type]} section, ' \
                   f'{self.position} mm'

    @lru_cache(100)
    def label_slice(self, position, channel):
        if self.view_idx == 0:
            view_slice = self.label[channel][position, :, :]
        elif self.view_idx == 1:
            view_slice = self.label[channel][:, position, :]
        else:
            view_slice = self.label[channel][:, :, position]
        return np.flipud(view_slice.T)

    def render(self, init=False):
        mapped_position, self.position = self.mapped_position(self.position)
        img_slice = self.img_slice(mapped_position)
        label_slice = None

        if self.label is not None:
            label_slice = self.label_slice(mapped_position, self.channel)

        if init:
            if self.show_text:
                self.axis.text(0.5, -0.1, self.legend, size=8, ha="center",
                               transform=self.axis.transAxes)

            self.axis_img = self.axis.imshow(img_slice, cmap='gray')

            if self.label is not None:
                self.axis_label = self.axis.imshow(
                    label_slice, cmap=self.cmap, alpha=self.alpha,
                    clim=[self.threshold, 1])

        if self.is_drawn:
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


class Figure:
    def __init__(self, dataset, subject_idx, views, view_org,
                 image_key_name, subject_org, update_all_on_scroll, add_text,
                 label_key_name, alpha, cmap, threshold, type2idx,
                 patch_sampler, nb_patches):
        self.dataset = dataset
        self.subject_idx = subject_idx
        self.views = views
        self.view_org = view_org
        self.image_key_name = image_key_name
        self.subject_org = subject_org
        self.update_all_on_scroll = update_all_on_scroll
        self.add_text = add_text
        self.label_key_name = label_key_name
        self.alpha = alpha
        self.cmap = cmap
        self.threshold = threshold
        self.type2idx = type2idx
        self.patch_sampler = patch_sampler
        self.nb_patches = nb_patches

        self.fig = None
        self.axes = None
        self.slider = None
        self.is_drawn = False
        self.view_objects = {}
        self.axes2view_keys = {}
        self.idx = None
        self.other_axes = []

    def set_attributes(self, fig, axes, slider, subject_org):
        self.fig = fig
        self.axes = axes
        self.slider = slider
        self.subject_org = subject_org

    @lru_cache(maxsize=10)
    def load_subjects(self):
        # DataLoader case
        if self.subject_idx is None:
            subjects = next(self.dataset)
            images, affines, labels = self.get_values_from_batch(subjects)
        else:
            subjects = [self.dataset[int(idx)] for idx in self.subject_idx]
            images, affines, labels = self.get_values_from_samples(subjects)

        self.idx = tuple(range(len(images)))
        self.adapt_subject_org()
        return images, affines, labels

    def get_values_from_batch(self, batch):
        if isinstance(batch, list):
            batch_len = len(batch[0][self.image_key_name]['data'])
        else:
            batch_len = len(batch[self.image_key_name]['data'])

        self.subject_idx = list(range(batch_len))
        images, affines, labels = [], [], None

        if self.label_key_name is not None:
            labels = []

        for i in range(batch_len):
            if isinstance(batch, list):
                batch_list = batch
            else:
                batch_list = [batch]
            for b in batch_list:
                images.append(b[self.image_key_name]['data'][i][0].numpy())
                affines.append(b[self.image_key_name]['affine'][i])

                if self.label_key_name is not None:
                    labels.append([b[key]['data'][i][0].numpy()
                                   for key in self.label_key_name])

        return images, affines, labels

    def get_values_from_samples(self, samples):
        images, affines, labels = [], [], None

        if self.label_key_name is not None:
            labels = []

        for sample in samples:
            if isinstance(sample, list):
                sample_list = sample
            else:
                sample_list = [sample]
            for s in sample_list:
                image = s[self.image_key_name]['data'][0].numpy()
                lab = None

                if self.label_key_name is not None:
                    lab = [s[key]['data'][0].numpy()
                           for key in self.label_key_name]

                if self.patch_sampler is not None:
                    image, lab = self.sample_patches(s, image, lab)

                images.append(image)
                affines.append(s[self.image_key_name]['affine'])

                if self.label_key_name is not None:
                    labels.append(lab)

        return images, affines, labels

    def sample_patches(self, sample, image, labels):
        for _ in range(self.nb_patches):
            patch = next(self.patch_sampler(sample))
            spatial_shape = np.array(patch.spatial_shape)
            i_ini, j_ini, k_ini = patch['index_ini']
            i_fin, j_fin, k_fin = patch['index_ini'] + spatial_shape
            im_patch = patch[self.image_key_name]['data'].numpy()[0].copy()

            im_patch[:2, :, :] = 1
            im_patch[-2:, :, :] = 1
            im_patch[:, :2, :] = 1
            im_patch[:, -2:, :] = 1
            im_patch[:, :, :2] = 1
            im_patch[:, :, -2:] = 1

            image[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = im_patch

            if self.label_key_name is not None:
                for label, key in zip(labels, self.label_key_name):
                    label_patch = patch[key]['data'].numpy()[0].copy()
                    label[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = label_patch

        return image, labels

    def adapt_subject_org(self):
        if np.prod(self.subject_org) < len(self.idx):
            self.subject_org = (1, len(self.idx))

    def set_views(self):
        indices = [(i, j) for i in range(self.view_org[0] * self.subject_org[0])
                   for j in range(self.view_org[1] * self.subject_org[1])]

        # Assign each view to its axis
        for i in self.idx:
            for j, view in enumerate(self.views):
                x = j // self.view_org[1] \
                    + i // self.subject_org[1] * self.view_org[0]
                y = j % self.view_org[1] \
                    + i % self.subject_org[1] * self.view_org[1]
                indices.remove((x, y))
                axis = self.axes[x, y]
                view_type, coordinate_system, _ = view
                view_key = (i, view_type, coordinate_system)

                if self.view_objects.get(view_key) is None:
                    self.view_objects[view_key] = View(
                        view, self.type2idx, i, self.fig, axis,
                        channel=0,
                        show_text=self.add_text,
                        cmap=self.cmap,
                        alpha=self.alpha,
                        threshold=self.threshold)
                    self.axes2view_keys[axis] = view_key

        for i, j in indices:
            self.other_axes.append(self.axes[i, j])

        for view_object in self.view_objects.values():
            view_object.set_is_drawn(self.is_drawn)

    def render_views(self, images, affines, labels, init=False):
        for i in self.idx:
            img, affine, label = images[i], affines[i], None

            if self.label_key_name is not None:
                label = labels[i]

            for view_key in filter(
                    lambda key: key[0] == i, self.view_objects.keys()):
                self.view_objects[view_key].set_img_affine_and_label(
                    img, affine, label
                )

        # Render each view
        for view_object in self.view_objects.values():
            view_object.axis.set_visible(True)
            view_object.render(init)

        # Update slider value
        if self.label_key_name is not None:
            self.on_slide(self.threshold)

        # Hide other axes
        for axis in self.other_axes:
            axis.set_visible(False)

    def display_figure(self):
        # Load subjects
        images, affines, labels = self.load_subjects()

        # Set views
        self.set_views()

        # Render views
        self.render_views(images, affines, labels, init=True)

    def set_is_drawn(self, is_drawn):
        self.is_drawn = is_drawn
        for view_object in self.view_objects.values():
            view_object.set_is_drawn(is_drawn)

    def clear_figure(self):
        # Clear View objects
        for view_key, view_object in self.view_objects.items():
            view_object.set_img_affine_and_label(None, None, None)

    def update(self, view_keys, update_function):
        for key in view_keys:
            view_object = self.view_objects[key]
            update_function(view_object)
            view_object.render()

        self.fig.canvas.flush_events()

    def on_scroll(self, event):
        view_key = self.axes2view_keys.get(event.inaxes)
        if view_key is not None:
            if event.button == 'down':
                delta = -1
            else:
                delta = 1

            if self.update_all_on_scroll:
                subject, view_type, coordinate_system = view_key
                view_keys = [(s, view_type, coordinate_system) for s in
                             self.idx]
            else:
                view_keys = [view_key]

            self.update(
                view_keys,
                lambda x: x.set_position(x.position + delta)
            )

    def on_key_press(self, event):
        if event.key == 'down':
            delta = -1
        else:
            delta = 1

        self.update(
            self.view_objects.keys(),
            lambda x: x.set_channel(
                (x.channel + delta) % len(self.label_key_name))
        )

    def on_slide(self, val):
        self.threshold = val
        self.slider.set_val(val)

        if self.is_drawn:
            for view_object in self.view_objects.values():
                view_object.threshold = self.threshold
                self.fig.canvas.restore_region(view_object.bg_cache)
                view_object.axis_label.set_clim([self.threshold, 1])
                view_object.axis.draw_artist(view_object.axis_img)
                view_object.axis.draw_artist(view_object.axis_label)

                self.fig.canvas.blit(view_object.axis.bbox)


class PlotDataset:
    """Draw an interactive plot of a few subjects from a torchio dataset.
    Scrolling on the different images enable to navigate sections.
    Hitting up and down keys enable to navigate label maps.
    Hitting pageup and pagedown keys enable to navigate figures.

    Args:
        dataset: a :py:class:`~torchio.ImagesDataset` or a
            :py:class:`~torch.data.utils.DataLoader` constructed from torchio.
        views: None or a sequence of views, each view is given as
            (view_type, coordinate_system, position),
            view_type is one among "sag", "ax" and "cor" which corresponds
            to sagittal, axial and coronal slices;
            coordinate_system is one among "vox" or "mm" and is responsible
            for placing the slice using position in term of voxel number
            or number of millimeters;
            position is either an integer between 0 and 100 if
            coordinate_system is "vox" or a float otherwise.
            If no value is provided, the default views are used:
            [['sag', 'vox', 50], ['ax', 'vox', 50], ['cor', 'vox', 50]].
        view_org: None or a sequence of length 2, responsible for the
            organisation of views in subplots. Default is (len(views), 1).
        image_key_name: a string that gives the key of the volume of interest
            in the dataset's samples.
        subject_idx: None, an integer or a sequence of integers. Defines
            which subjects are plotted.
            If subject_idx is a sequence of integers, it is directly used
            as the list of indexes,
            if subject_idx is an integer, subjects_idx subjects are taken at
            random in the dataset.
            Finally, if subject_idx is None, all subjects are taken.
            Default value is 5.
        subject_org: None or a sequence of length 2, responsible for the
            organisation of subjects in subplots.
            Default is (1, len(subject_idx)).
        figsize: Sequence of length 2, size of the figure.
        update_all_on_scroll: bool, if True all views with the same view_type
            and coordinate_system are updated when scrolling on one of them.
            Doing so supposes that they all have the same shape.
            Default is False.
        add_text: Boolean to choose if you want the axis legend to be printed.
            Default is True.
        label_key_name: a string or a list of strings that gives the key(s) of
            the label maps of interest in the dataset's samples.
        alpha: overlay opacity, used when plotting label, default is 0.2.
        cmap: the colormap used to plot label, default is 'RdBu'.
        patch_sampler: a sampler used to generate patches from the volumes,
            if sampler is not None, the patches are superimposed on the
            volumes with white borders. Default is None.
        nb_patches: the number of patches to draw from each sample.
            Used only if patch_sampler is not None.
        threshold: the threshold below which label are not shown.
        preload: Boolean to choose whether to preload all subjects are not.
            Default is True.
    """

    def __init__(self, dataset, views=None, view_org=None,
                 image_key_name='image', subject_idx=5, subject_org=None,
                 figsize=(16, 9), update_all_on_scroll=False, add_text=True,
                 label_key_name=None, alpha=0.2, cmap='RdBu',
                 patch_sampler=None, nb_patches=4, threshold=0.01,
                 preload=True):
        self.dataset = dataset
        self.views = views if views is not None else vox_views
        self.view_org = self.parse_view_org(view_org)
        self.image_key_name = image_key_name
        self.subject_idx = self.parse_subject_idx(subject_idx)
        self.subject_org = self.parse_subject_org(subject_org)
        self.figsize = figsize
        self.update_all_on_scroll = update_all_on_scroll
        self.add_text = add_text
        self.label_key_name = label_key_name
        if isinstance(self.label_key_name, str):
            self.label_key_name = [self.label_key_name]
        self.alpha = alpha
        self.cmap = cm.get_cmap(cmap)
        self.cmap.set_under(color='k', alpha=0)
        self.patch_sampler = patch_sampler
        self.nb_patches = nb_patches
        self.threshold = threshold
        self.preload = preload

        self.figure_objects = []
        self.fig = None
        self.axes = None
        self.slider = None
        self.current_figure = 0
        self.updating_views = False

        self.coordinate_system_list = ["vox", "mm"]
        self.type2idx = ["sag", "cor", "ax"]

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
            valid = reduce(lambda acc, val: acc and 0 <= val < data_len,
                           subject_idx, True)
            if not valid:
                raise ValueError('Invalid index sequence')
            return subject_idx
        elif isinstance(subject_idx, int):
            return np.random.choice(range(data_len), min(data_len, subject_idx),
                                    replace=False).astype(int)
        else:
            return list(range(data_len))

    def parse_subject_org(self, subject_org):
        if isinstance(subject_org, Sequence) and len(subject_org) == 2:
            return subject_org
        else:
            return 1, len(self.subject_idx)

    def check_views(self):
        for view_type, coordinate_system, _ in self.views:
            if coordinate_system not in self.coordinate_system_list:
                raise ValueError(
                    f'coordinate_system {coordinate_system} not recognized '
                    f'among {self.coordinate_system_list}'
                )
            if view_type not in self.type2idx:
                raise ValueError(f'view_type {view_type} not recognized '
                                 f'among {self.type2idx}')

    def create_subplot(self):
        subplot_shape = (self.view_org[0] * self.subject_org[0],
                         self.view_org[1] * self.subject_org[1])

        self.fig, axes = plt.subplots(*subplot_shape, figsize=self.figsize)
        self.fig.tight_layout()
        self.axes = axes.reshape(subplot_shape)

        # Add sliders to set overlay display threshold
        if self.label_key_name is not None:
            axis = plt.axes([0.01, 0.05, 0.02, 0.9])
            self.slider = NewSlider(
                axis, 'Threshold', 0, 1, valinit=self.threshold,
                valstep=0.01, orientation='vertical')

        # Add event handlers
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_press_event', self.on_go_to)

        if self.label_key_name is not None:
            self.slider.on_changed(self.on_slide)

        # Remove axis for all subplots
        for axis in self.axes.ravel():
            axis.axis('off')

        # Remove space between subplots
        if self.label_key_name is None:
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0,
                                hspace=0)
        else:
            plt.subplots_adjust(left=0.04, right=0.96, bottom=0, top=1,
                                wspace=0, hspace=0)

        # Add colorbar
        if self.label_key_name is not None:
            # Create a fake label to make the colorbar invariant
            # to threshold changes
            axis = plt.axes([0., 0., 0., 0.])
            fake_label = axis.imshow(np.linspace(0, 1, 100).reshape(10, 10),
                                     cmap=self.cmap, alpha=self.alpha)
            fake_label.set_visible(False)
            color_bar_axis = self.fig.add_axes([0.96, 0.05, 0.02, 0.9])
            self.fig.colorbar(fake_label, cax=color_bar_axis)

    def init_plot(self):
        nb_subject_per_figure = np.product(self.subject_org)
        nb_figures = math.ceil(len(self.subject_idx) / nb_subject_per_figure)

        # Create figure objects
        for i in range(nb_figures):
            if isinstance(self.dataset, DataLoader):
                subject_idx = None
            else:
                subject_idx = self.subject_idx[
                    i * nb_subject_per_figure: (i + 1) * nb_subject_per_figure
                ]
            self.figure_objects.append(
                Figure(self.dataset, subject_idx, self.views, self.view_org,
                       self.image_key_name, self.subject_org,
                       self.update_all_on_scroll, self.add_text,
                       self.label_key_name, self.alpha, self.cmap,
                       self.threshold, self.type2idx, self.patch_sampler,
                       self.nb_patches)
            )

        # Preload subjects
        if self.preload:
            for figure_object in self.figure_objects:
                figure_object.load_subjects()
        else:
            self.figure_objects[0].load_subjects()
        self.subject_org = self.figure_objects[0].subject_org

        # Create matplotlib figure and axes
        self.create_subplot()

        # Set figure and axes
        for figure_object in self.figure_objects:
            figure_object.set_attributes(
                self.fig, self.axes, self.slider, self.subject_org)

        # Display first figure object
        self.figure_objects[0].display_figure()

        # Draw canvas
        self.fig.canvas.draw()
        self.figure_objects[0].set_is_drawn(True)

    def on_scroll(self, event):
        if not self.updating_views:
            self.updating_views = True
            self.figure_objects[self.current_figure].on_scroll(event)
            self.updating_views = False

    def on_key_press(self, event):
        if self.label_key_name is not None and not self.updating_views:
            if event.key == 'down' or event.key == 'up':
                self.updating_views = True
                self.figure_objects[self.current_figure].on_key_press(event)
                self.updating_views = False

    def on_go_to(self, event):
        if not self.updating_views:
            if event.key == 'pageup' or event.key == 'pagedown':
                self.updating_views = True
                if event.key == 'pageup':
                    delta = -1
                else:
                    delta = 1

                num = (self.current_figure + delta) % len(self.figure_objects)

                self.figure_objects[self.current_figure].clear_figure()
                self.figure_objects[num].set_is_drawn(True)
                self.figure_objects[num].display_figure()
                self.current_figure = num
                self.fig.canvas.resize_event()
                self.updating_views = False

    def on_slide(self, val):
        if not self.updating_views:
            self.updating_views = True
            self.figure_objects[self.current_figure].on_slide(val)
            self.fig.canvas.resize_event()
            self.updating_views = False


class NewSlider(Slider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bg_cache = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)

    def set_val(self, val):
        """
        Set slider value to *val*

        Parameters
        ----------
        val : float
        """
        xy = self.poly.xy
        if self.orientation == 'vertical':
            xy[1] = 0, val
            xy[2] = 1, val
        else:
            xy[2] = val, 1
            xy[3] = val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % val)

        if self.drawon and self.ax.figure._cachedRenderer is None:
            self.ax.figure.canvas.draw_idle()
        elif self.drawon:
            self.ax.figure.canvas.restore_region(self.bg_cache)
            self.ax.draw_artist(self.poly)
            self.ax.draw_artist(self.valtext)
            self.ax.figure.canvas.blit()

        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(val)
