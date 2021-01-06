import os
import dash
import torch
import torchio
import numpy as np
import pandas as pd
import nibabel as nb
from pathlib import PosixPath
import matplotlib.pyplot as plt
from os.path import join as opj
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from torchio import ScalarImage, LabelMap, Subject


class GZReader:
    def __init__(self, gz_path=None, df_data=None, out_tmp=""):
        self.gz_path = gz_path
        self.df_data = None
        self.out_tmp = out_tmp
        if gz_path:
            self.open(gz_path=gz_path)
        if df_data is not None:
            self.df_data = df_data
        self.dash_app = None
        self.written_files = []

    def open(self, gz_path):
        if isinstance(gz_path, list):
            df_list = []
            for one_csv in gz_path:
                df_list.append(pd.read_pickle(one_csv))
            self.df_data = pd.concat(df_list, sort=False).reindex()
        else:
            self.df_data = pd.read_pickle(gz_path)

    def close(self):
        del self.df_data
        self.gz_path = None
        self.df_data = None

    def get_row(self, idx):
        return self.df_data.iloc[idx]

    def read_path(self, path):
        if isinstance(path, list):
            return [self.read_path(p) for p in path]

        elif isinstance(path, PosixPath):
            return opj(str(path))

        elif isinstance(path, str):
            try:
                eval_path = eval(path)
                return self.read_path(eval_path)
            except Exception:
                return opj(path)

        else:
            raise TypeError("Could not read path: {}".format(path))

    def get_volume_nibabel(self, idx, return_orig=False):
        subject_row = self.get_row(idx)
        subject_path = self.read_path(subject_row["image_filename"])
        if return_orig:
            volume = nb.load(subject_path)
        else:
            tio_data = self.get_volume_torchio(idx, return_orig=return_orig)
            tio_data = tio_data["t1"] if "t1" in tio_data.keys() else tio_data["image_from_labels"]
            data, affine = tio_data["data"], tio_data["affine"]
            volume = nb.Nifti1Image(data, affine)
        return volume

    def extract_from_history(self, col, key, save_csv=False, col_name=None):
        data_col = self.df_data[~self.df_data[col].isnull()][col]
        dict_data = data_col.apply(lambda x: eval(x)[key])
        if save_csv:
            if not col_name:
                col_name = key
            self.df_data[col_name] = dict_data
            self.df_data.to_csv(self.gz_path)
        return dict_data

    def check_dash(self):
        if not self.dash_app:
            self.dash_app = dash.Dash()

    def get_volume_torchio(self, idx, return_orig=False):
        subject_row = self.get_row(idx)
        dict_suj = dict()
        if not pd.isna(subject_row["image_filename"]):
            path_imgs = self.read_path(subject_row["image_filename"])
            if path_imgs:
                if isinstance(path_imgs, list):
                    imgs = ScalarImage(tensor=np.asarray([nb.load(p).get_fdata() for p in path_imgs]))
                else:
                    imgs = ScalarImage(path_imgs)
                dict_suj["t1"] = imgs

        if "label_filename" in subject_row.keys() and not pd.isna(subject_row["label_filename"]):
            path_imgs = self.read_path(subject_row["label_filename"])
            if isinstance(path_imgs, list):
                imgs = LabelMap(tensor=np.asarray([nb.load(p).get_fdata() for p in path_imgs]))
            else:
                imgs = LabelMap(path_imgs)
            dict_suj["label"] = imgs
        sub = Subject(dict_suj)
        if "history" not in self.df_data.columns:
            return sub
        else:
            trsfms = self.get_transformations(idx)
            res = sub
            for tr in trsfms.transform.transforms:
                if isinstance(tr, torchio.transforms.LabelsToImage):
                    tr.label_key = "label"
                if isinstance(tr, torchio.transforms.MotionFromTimeCourse):
                    output_path = opj(self.out_tmp, "{}.png".format(idx))
                    fitpars = tr.fitpars["t1"]
                    plt.figure()
                    plt.plot(fitpars.T)
                    plt.legend(["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"])
                    plt.xlabel("Timesteps")
                    plt.ylabel("Magnitude")
                    plt.title("Motion parameters")
                    plt.savefig(output_path)
                    plt.close()
                    self.written_files.append(output_path)
                res = tr(res)
            res = trsfms(sub)
            return res

    def get_transformations(self, idx):
        from torchio.transforms import Compose
        import torchio.transforms

        row = self.get_row(idx)
        trsfms_order = row["history"] #[r for r in row["transfo_order"].split("_") if r != ""]
        trsfm_composition = Compose(trsfms_order)
        return trsfm_composition

    def plot_hist(self, data, save=None):
        if isinstance(data, nb.Nifti1Image):
            data = data.get_fdata().reshape(-1)
        elif isinstance(data, torch.Tensor):
            data = data.flatten().numpy()
        n, bins, patches = plt.hist(data, bins=256, range=(1, data.max()), facecolor='red', alpha=0.75,
                                    histtype='step')
        if save:
            plt.savefig(save)
        plt.close()

    def scatter(self, col_x, col_y, renderer="browser", color=None, port_number=8050, **kwargs):
        fig = go.Figure()
        filtered_df = self.df_data[~self.df_data[col_x].isnull() & ~self.df_data[col_y].isnull()]
        if not color or color not in self.df_data.columns:
            fig.add_trace(go.Scatter(x=filtered_df[col_x], y=filtered_df[col_y],
                                     hovertext=filtered_df["image_filename"], text=filtered_df.index.to_numpy(),
                                     mode="markers", **kwargs))
        else:
            categories = filtered_df[color].unique().astype(str)
            traces = []
            for idx, cat in enumerate(categories):
                cat_data = filtered_df[filtered_df[color] == cat]
                traces.append(go.Scatter(x=cat_data[col_x], y=cat_data[col_y], marker_symbol=idx,
                                         hovertext=cat_data["image_filename"], text=cat_data.index.to_numpy(),
                                         mode="markers", name=cat, **kwargs))
            fig.add_traces(traces)
        fig.update_layout(xaxis_title=col_x,
                          yaxis_title=col_y,
                          legend=dict(
                              orientation="h",
                              yanchor="bottom",
                              y=1.02,
                              xanchor="right",
                              x=1
                          )
                          )
        self.check_dash()
        self.dash_app.layout = html.Div(children=[
            html.H1(children='CSV MRI Scatter Plot'),

            html.Div(children='''
                Plot from {}
            '''.format(self.gz_path)),

            dcc.Graph(
                id='scatter-plot',
                figure=fig
            ),
            html.Div(id='output-click'),
        ])

        @self.dash_app.callback(
            [Output('output-click', 'children'),],
            [Input('scatter-plot', 'clickData'),],
        )
        def display_click_data(clickData):
            path = clickData["points"][0]["hovertext"]
            idx = clickData["points"][0]["text"]
            out_path = opj(self.out_tmp, str(idx) + ".nii")
            if not os.path.exists(out_path):
                transformed = self.get_volume_torchio(idx)
                key = list(transformed.get_images_dict(intensity_only=True).keys())[0]
                transformed = transformed[key]
                data, affine = transformed['data'].squeeze().numpy(), transformed["affine"]
                nib_volume = nb.Nifti1Image(data, affine)
                nib_volume.to_filename(out_path)
                self.written_files.append(out_path)
                self.plot_hist(nib_volume, save=opj(self.out_tmp, str(idx) + "_hist.png"))
            if path:
                os.system("mrviewv " + out_path)
            return "Viewing: {}".format(path)

        self.dash_app.run_server(debug=False, port=port_number)


