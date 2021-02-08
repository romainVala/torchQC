import json
import torch
from torchio import Subject, LabelMap, ScalarImage
import torchio
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import os
import pandas as pd
import inspect
import nibabel as nb
import numpy as np
from pathlib import PosixPath
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.interactive(True)
from os.path import join as opj
from nibabel.viewers import OrthoSlicer3D as ov


def default_json_str_to_eval_python(x):
    if not isinstance(x,str):
        return x
    if pd.isna(x):
        return None

    x = x.replace('true', 'True')
    x = x.replace('false', 'False')
    x = x.replace('null', 'None')
    x = x.replace("<","'<")
    x = x.replace(">", ">'")  #when python function are print in csv so transform as str
    #same with array, more difficult
    ind_array = x.find('array')
    while ind_array>0:
        ind_next_tag = x[ind_array:].find("'") #suposing next tag
        cut_int = ind_array + ind_next_tag
        #x = x[:ind_array] + "'" + x[ind_array:cut_int] + "', " + x[cut_int:] #make the array a str
        #let's just remove it
        x = x[:ind_array] + "'aray_removed', " + x[cut_int:]
        ind_array = x.find('array')

    x = eval(x)
    if isinstance(x,str):
        x = eval(x)
    return x


class ModelCSVResults(object):

    def __init__(self, csv_path=None, df_data=None, out_tmp=""):
        self.csv_path = csv_path
        self.df_data = None
        self.out_tmp = out_tmp
        if csv_path:
            self.open(csv_path=csv_path)
        if df_data is not None:
            self.df_data = df_data
        self.dash_app = None
        self.written_files = []

    def open(self, csv_path):
        if isinstance(csv_path, list):
            df_list = []
            for one_csv in csv_path:
                df_list.append(pd.read_csv(one_csv))
            self.df_data = pd.concat(df_list, sort=False).reindex()
        else:
            self.df_data = pd.read_csv(csv_path)

    def close(self):
        del self.df_data
        self.csv_path = None
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
            tio_data = tio_data["volume"] if "volume" in tio_data.keys() else tio_data["image_from_labels"]
            data, affine = tio_data["data"], tio_data["affine"]
            volume = nb.Nifti1Image(data, affine)
        return volume

    def get_volume_torchio(self, idx, return_orig=False):
        subject_row = self.get_row(idx)
        dict_suj = dict()
        if not pd.isna(subject_row["image_filename"]):
            path_imgs = self.read_path(subject_row["image_filename"])
            if isinstance(path_imgs, list):
                imgs = ScalarImage(tensor=np.asarray([nb.load(p).get_fdata() for p in path_imgs]))
            else:
                imgs = ScalarImage(path_imgs)
            dict_suj["volume"] = imgs

        if "label_filename" in subject_row.keys() and not pd.isna(subject_row["label_filename"]):
            path_imgs = self.read_path(subject_row["label_filename"])
            if isinstance(path_imgs, list):
                imgs = LabelMap(tensor=np.asarray([nb.load(p).get_fdata() for p in path_imgs]))
            else:
                imgs = LabelMap(path_imgs)
            dict_suj["label"] = imgs
        sub = Subject(dict_suj)
        if return_orig or "transfo_order" not in self.df_data.columns:
            return sub
        else:
            trsfms, seeds = self.get_transformations(idx)
            for tr in trsfms.transform.transforms:
                if isinstance(tr, torchio.transforms.RandomLabelsToImage):
                    tr.label_key = "label"
                if isinstance(tr, torchio.transforms.RandomMotionFromTimeCourse):
                    output_path = opj(self.out_tmp, "{}.png".format(idx))
                    if "fitpars" in self.df_data.columns:
                        fitpars = np.loadtxt(self.df_data["fitpars"][idx])
                        tr.fitpars = fitpars
                        tr.simulate_displacement = False
                    else:

                        res = sub
                        for trsfm, seed in zip(trsfms.transform.transforms , seeds):
                            if seed:
                                res = trsfm(res, seed)
                            else:
                                res = trsfm(res)
                        del res
                        fitpars = tr.fitpars
                    plt.figure()
                    plt.plot(fitpars.T)
                    plt.legend(["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"])
                    plt.xlabel("Timesteps")
                    plt.ylabel("Magnitude")
                    plt.title("Motion parameters")
                    plt.savefig(output_path)
                    plt.close()
                    self.written_files.append(output_path)
            res = sub
            for trsfm, seed in zip(trsfms.transform.transforms, seeds):
                if seed:
                    res = trsfm(res, seed)
                else:
                    res = trsfm(res)
            #res = trsfms(sub, seeds)
            return res

    def display_original_data(self, idx):
        volume = self.get_volume_nibabel(idx)
        ov(volume.get_data())

    def trsfm_arg_eval(self, arg_to_eval):
        from torchio.transforms.preprocessing.intensity.normalization_transform import NormalizationTransform

        if isinstance(arg_to_eval, str):
            try:
                if arg_to_eval.startswith("<function"):
                    arg_to_eval = arg_to_eval.split()[1]
                return eval(arg_to_eval)

            except NameError:
                return arg_to_eval
        return arg_to_eval

    def get_transformations(self, idx):
        from torchio.transforms import Compose
        import torchio.transforms

        row = self.get_row(idx)
        trsfms_order = [r for r in row["transfo_order"].split("_") if r != ""]
        trsfm_list = []
        trsfm_seeds = []
        for trsfm_name in trsfms_order:
            if trsfm_name not in ["OneOf"]:
                trsfm_history = default_json_str_to_eval_python(row["T_"+trsfm_name])
                trsfm = getattr(torchio.transforms, trsfm_name)
                #trsfm_seed = trsfm_history["seed"] if "seed" in trsfm_history.keys() else None
                if trsfm_name == "RandomMotionFromTimeCourse":
                    trsfm_seeds.append(trsfm_history["seed"])
                    del trsfm_history["seed"]
                    init_args = inspect.getfullargspec(trsfm.__init__).args
                    print(init_args)
                    trsfm_history = {hist_key: self.trsfm_arg_eval(hist_val)
                                     for hist_key, hist_val in trsfm_history.items()
                                     if hist_key in init_args and hist_key not in ['metrics', 'fitpars', "read_func"]}
                else:
                    trsfm_seeds.append(None)

                if trsfm_name == "RescaleIntensity":
                    trsfm_history["masking_method"] = None #self.trsfm_arg_eval(str(trsfm_history["masking_method"]))
                #if "seed" in trsfm_history.keys():
                #    del trsfm_history["seed"]
                print(f"Found transform: {trsfm_name}\n{trsfm_history}")

                trsfm_history = {k: v for k, v in trsfm_history.items() if k not in ["probability"]}
                trsfm = trsfm(**trsfm_history)
                #init_args = inspect.getfullargspec(trsfm.__init__).args
                """
                hist_kwargs_init = {hist_key: self.trsfm_arg_eval(hist_val)
                                    for hist_key, hist_val in trsfm_history.items()
                                    if hist_key in init_args and hist_key not in ['metrics', 'fitpars', "read_func"]}

                trsfm = trsfm(**hist_kwargs_init)
                """
                trsfm_list.append(trsfm)
        trsfm_composition = Compose(trsfm_list)
        return trsfm_composition, trsfm_seeds

    def normalize_dict_to_df(self, col, suffix=None, eval_func=default_json_str_to_eval_python):
        if isinstance(col, list):
            if not isinstance(suffix, list):
                suffix = [suffix for _ in col]
            for one_col, one_suffix in zip(col, suffix):
                if one_col not in self.df_data:
                    print('WARNING col {} is missing'.format(one_col))
                else:
                    df = self.normalize_dict_to_df(one_col, suffix=one_suffix, eval_func=eval_func)
            return df

        if suffix is None:
            suffix = col
        dict_vals = self.df_data[col]
        if eval_func:
            dict_vals = self.df_data[col].apply(eval_func)
        #print(dict_vals[~pd.isna(dict_vals)])
        if isinstance(dict_vals[0], list):
            dict_vals = dict_vals.apply(lambda x: x[0]) #BAD what if more ...
        if isinstance(dict_vals[0],tuple):  #tupe 0 is transfo name RamdomMotionFRomTimeCourse
            dict_vals = dict_vals.apply(lambda x: x[1])  # BAD what if more ...

        val_names = dict_vals[~pd.isna(dict_vals)].iloc[0].keys()
        for name in val_names:
            added_key_name = f"{suffix}_{name}" if suffix else f"{name}"
            self.df_data[added_key_name] = dict_vals.apply(lambda x: x[name] if not(pd.isna(x)) else None)
        return self.df_data


    def extract_from_history(self, col, key, save_csv=False, col_name=None):
        data_col = self.df_data[~self.df_data[col].isnull()][col]
        dict_data = data_col.apply(lambda x: eval(x)[key])
        if save_csv:
            if not col_name:
                col_name = key
            self.df_data[col_name] = dict_data
            self.df_data.to_csv(self.csv_path)
        return dict_data

    def check_dash(self):
        if not self.dash_app:
            self.dash_app = dash.Dash()

    def clean_tmp_dir(self):
        for f in self.written_files:
            os.remove(f)

    def correlation(self, col_x, col_y):
        filtered_df = self.df_data[~self.df_data[col_x].isnull() & ~self.df_data[col_y].isnull()]
        return filtered_df[col_y].corr(filtered_df[col_x])

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
            '''.format(self.csv_path)),

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
