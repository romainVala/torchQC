# Segmentation

This package aims to provide a generic code based on configuration files
to train a Pytorch neural network on a 3D segmentation task.

## Configuration files
A few configuration files are needed to run the code and should be referenced
in the `main.json` file (the name of the file is not important).

#### main.json
This file references all the configuration files needed to run the code.
```json
{
    "data": "some_path/data.json",
    "transform": "some_path/transform.json",
    "model": "some_path/model.json",
    "run": "some_path/run.json"
}
```
`"data"`, `"transform"`, `and `"model"` are mandatory. Depending on the mode the program is run
in, one additional key is needed:
- `"run"` if mode is `"train"`,`"eval"` or`"infer"`,
- `"visualization"` if mode is`"viz"` (not implemented yet).

#### data.json
This file tells the code where to find the data to use in the segmentation task and the repartition
between train, validation and test data. If `"subject_shuffle"` is `true`, subjects are shuffled
before being assigned to train, validation and test sets. <br>
Either patterns using unix regular expressions or explicit paths can be used to list subjects.
Both can be used at the same time (patterns are read first). <br>
This file also defines the behaviour of the `DataLoader`, using patches or not, the 
number of workers used to load data (`-1` means all available workers are used) and 
the batch size. Furthermore, it sets which keys will be used to retrieve images and labels 
from subjects. The `"labels"` key defines which channels from the targets will be used.
```json
{
    "images": 
    {
        "t1": {"type": "intensity", "components":  ["t1"]},
        "label": {"type":  "label", "components":  ["GM", "CSF", "WM"]}
    },
    "patterns": 
    [
        {
            "root": "/path_to_t1/*/",
            "name_pattern": "([0-9]+)", 
            "components": {"t1": {"path": "T1w/T1w_acpc_dc.nii.gz", "image": "t1"}}
        }, 
        {
            "root": "/path_to_label/suj_*/", 
            "name_pattern": "([0-9]+)", 
            "components": {
                "GM": {"path": "grey.nii.gz", "image":  "label"},
                "CSF": {"path": "csf.nii.gz", "image":  "label"},
                "WM": {"path": "white.nii.gz", "image": "label"}
            }
        }
    ], 
    "repartition": [0.7, 0.15, 0.15],
    "subject_shuffle": true,
    "subject_seed": 0,
    "image_key_name": "t1",
    "label_key_name": "label",
    "labels": ["GM"],
    "batch_size": 2,
    "num_workers": 0,
    "queue": 
    {
        "attributes": 
        {
            "max_length": 8, 
            "samples_per_volume": 4
        }, 
        "sampler": 
        {
            "name": "LabelSampler",
            "module": "torchio.data.sampler",
            "attributes": 
            {
                "patch_size": 64,
                "label_name": "grey",
                "label_probabilities": {"0": 0.1, "1": 0.9}
            }
        }
    },
    "collate_fn": 
    {
        "name": "history_collate",
        "module": "segmentation.collate_functions"
    },
    "batch_shuffle": true,
    "batch_seed": 0
}
```

```json
{
    "images": 
    {
        "t1": {"type": "intensity", "components": ["t1"]},
        "label": {"type": "label", "components": ["GM", "CSF", "WM"]}
    },
    "paths": 
    [
        {
            "name": "subject1",
            "components": 
            {
                "t1": {"path": "/path_to_t1_for_subject1/T1w/T1w_acpc_dc.nii.gz", "image": "t1"},
                "GM": {"path": "/path_to_label_for_subject1/grey.nii.gz", "image": "label"},
                "CSF": {"path": "/path_to_label_for_subject1/csf.nii.gz", "image": "label"},
                "WM": {"path": "/path_to_label_for_subject1/white.nii.gz", "image": "label"}
            }
        }, 
        {
            "name": "subject2",
            "components": 
            {
                "t1": {"path": "/path_to_t1_for_subject2/T1w/T1w_acpc_dc.nii.gz", "image": "label"},
                "grey": {"path": "/path_to_label_for_subject2/grey.nii.gz", "image":  "label"},
                "csf": {"path": "/path_to_label_for_subject2/csf.nii.gz", "image":  "label"},
                "white": {"path": "/path_to_label_for_subject2/white.nii.gz", "image":  "label"}
            }
        }
    ], 
    "repartition": [0.7, 0.15, 0.15],
    "subject_shuffle": true,
    "subject_seed": 0,
    "image_key_name": "t1",
    "label_key_name": "label",
    "labels": ["GM"],
    "batch_size": 2,
    "num_workers": 0,
    "batch_shuffle": true,
    "batch_seed": 0
}
```
`"images"`, `"batch_size"`, `"image_key_name"` and `"label_key_name"` are mandatory. 
If `"patterns"` are not empty, each pattern must have keys `"root"` and `"images"`.
If `"paths"` are not empty, each path must have keys `"name"` and `"images"`. 
If `"queue"` is not empty, it must have the attribute `"sampler"` which must have keys
`"name"`, `"module"` and `"attributes"`. In such case, `"attributes"` must define the
attribute `"patch_size"`.


If a component is missing from a subject's image, the subject is discarded with a warning.

#### transform.json
This file defines which transforms (preprocessing and data augmentation) are applied to train,
validation and test samples. The same transforms are applied to validation and test samples. There is the option of 
adding the computation of a metric between the data before and after a specific transform by adding the attribute 
`"metrics"` and setting the value of `"compare_to_original"` to 1.
```json
{
    "save": false,
    "train_transforms": 
    [
        {
            "name": "RandomBiasField",
            "attributes": {
                "compare_to_original": 1,
                "metrics": [
                    {
                        "L1": {
                            "wrapper": {
                                "attributes": {
                                    "metric_func": {
                                        "attributes": {},
                                        "module": "torch.nn",
                                        "name": "L1Loss"
                                    },
                                    "metric_name": "L1"
                                },
                                "type": "metricwrapper"
                            }
                        }
                    },{
                        "SSIM": {
                            "attributes": {
                                "average_method": "mean",
                                "mask_keys": [
                                    "brain"
                                ]
                            },
                            "module": "torchio.metrics",
                            "name": "SSIM3D"
                        }
                    }
                ]
            }
        },
        {
            "name": "RandomNoise"
        }, 
        {
            "name": "RandomFlip",
            "attributes": {"axes": [0]}
        },
        {
            "name": "OneOf", 
            "is_selection": "True",
            "transforms":
            [
                {
                    "prob": 0.8, 
                    "transform":
                    {
                        "name":"RandomAffine"
                    }
                }, 
                {
                    "prob": 0.2,
                    "transform":
                    {
                        "name":"RandomElasticDeformation"
                    }
                }
            ]
        }
    ], 
    "val_transforms": []
}
```
`"train_transforms"` and `"val_transforms"` are mandatory.

If `"save"` is `true`, when doing whole image evaluation, transformed
images will be save to the result directory.

#### model.json
This file defines the model to use, its parameters and if it is loaded from a saved model.
If attribute `"last_one"` is `true`, it means that last saved model will be used. If no model
has been saved so far, a new model will be used. To use a model saved at a specific path,
`"last_one"` must be set to `false` and `"path"` must be filled with the path to the model.
```json
{
    "module": "unet", 
    "name": "UNet3D", 
    "attributes": 
    {
        "in_channels": 1, 
        "out_classes": 1, 
        "padding": 1, 
        "residual": true, 
        "num_encoding_blocks": 5, 
        "encoder_out_channel_lists": 
        [
            [30, 30, 30], 
            [60, 60, 60], 
            [120, 120, 120], 
            [240, 240, 240], 
            [320, 320, 320]
        ], 
        "decoder_out_channel_lists":
        [
            [240, 240, 240], 
            [120, 120, 120],
            [60, 60, 60],
            [30, 30, 30]
        ]
    },
    "last_one": true,
    "device": "cuda"
}
```

```json
{
    "module": "unet",
    "name": "UNet3D", 
    "last_one": false,
    "path": "/path_to_model"
}
```
`"name"` and `"module"` are mandatory.

#### run.json
This file defines
- the losses used to train the model (`"criteria"`),
- the optimizer (`"optimizer"`),
- the frequency at which logs are printed (`"log_frequency"`),
- how often the performances of the model are saved and which method is used for that (`"save"`),
- how often the model is evaluated on the validation set and which metrics are used to assess the 
quality of the segmentation on the validation set (`"validation"`). The model 
is saved after every evaluation loop on the validation set,
- how often inference on whole images is done, this is relevant only if patches are
 used during training (`"whole_image_inference_frequency"` in `"validation"`, this frequency is
 in number of epochs),
- the number of epochs,
- which methods are used to get tensors from data.
```json
{
    "criteria": 
    [
        {
            "module": "segmentation.losses.dice_loss", 
            "name": "Dice",
            "method": "mean_dice_loss",
            "weight": 1
        }
    ], 
    "optimizer": 
    {
        "module": "torch.optim", 
        "name": "Adam", 
        "attributes": {"lr": 0.0001},
        "lr_scheduler": 
        {
            "module": "torch.optim.lr_scheduler",
            "name": "ReduceLROnPlateau",
            "attributes": 
            {
                "mode": "min",
                "patience": 6
            }
        }
    }, 
    "save": 
    {
        "record_frequency": 10,
        "batch_recorder": "record_segmentation_batch",
        "prediction_saver": "save_volume"
    }, 
    "validation": 
    {
        "whole_image_inference_frequency": 100, 
        "patch_overlap": 8, 
        "eval_frequency": 100, 
        "reporting_metrics": 
        [
            {
                "module": "segmentation.losses.dice_loss", 
                "name": "Dice",
                "method": "mean_dice_loss"
            },
            {
                "module": "segmentation.metrics.overlap_metrics",
                "name": "OverlapMetric",
                "method": "mean_false_positives",
                "mask": "white",
                "reported_name": "FP_grey_in_white",
                "channels": ["grey"]
            }
        ]
    }, 
    "seed": 0, 
    "log_frequency": 10,
    "n_epochs": 500,
    "data_getter": "get_segmentation_data"
}
```
`"criteria"`, `"optimizer"`, `"save"`, `"validation"` and `"n_epochs"` are mandatory.

In the `"optimizer"` dictionary, `"name"` and `"module"` are mandatory. <br>
If `"lr_scheduler"` is present in the `"optimizer"` dictionary, it must 
have `"name"` and `"module"`.
In the `"save"` dictionary, `"record_frequency"` is mandatory. <br>

#### visualization.json
This file defines the visualization parameters. It uses `PlotDataset` to make plots.
```json
{
    "kwargs": 
    {
        "subject_idx": [0, 1, 2, 3, 4], 
        "update_all_on_scroll": true,
        "nb_patches": 4
    },
    "set": "val"
}
```

#### extra_file.json
This file can overwrites `data.json`, `transform.json`, `model.json` and/or the results directory.
```json
{
    "transform": 
    {
        "train_transforms": [],
        "val_transforms": 
        [
            {
                "name": "RandomBiasField"
            }, 
            {
                "name": "RescaleIntensity", 
                "attributes": 
                {
                    "out_min_max": [0, 1],
                    "percentiles": [0.5, 99.5]
                }
            }
        ]
    },
    "results_dir": "eval_with_bias_field"
}
```

## Run the program
The main entry point of the program is the `segmentation_pipeline.py` file.
Therefore, in order to run the program, one can run the following command:
```shell script
python ./segmentation/segmentation_pipeline.py -f '/path_to_main_config_file/main.json' -r '/results_dir'
```
`-f` is the path to the main json configuration file.

`-r` is the path to the results directory. All saved files are saved to this directory,
including log files.

The `--mode` argument can take the following values:
- `"train"` (default value): the selected model is trained on training data and evaluated
on validation data;
- `"eval"`: the selected model is evaluated on validation data;
- `"infer"`: the selected model is used to make predictions on test data;
- `"visualization"`: a visualization of the data and eventually the predictions made by the
model is shown.

The `--debug` argument's default value is `0`, if a different value is given, debug information
will be printed in the console.

The `--safe_mode` argument's default value is `False`, if `True`, the user is asked if he
wants to proceed before overwriting an existing configuration file in the results directory.

The `--viz` argument is only used when `--mode` is `"visualization"`. Values ranging from
0 to 5 are accepted. Default value is `0`.
- If value is `0`: volumes are shown,
- If value is `1`: volumes with labels are shown,
- If value is `2`: volumes with patches are shown,
- If value is `3`: volumes with patches and labels are shown,
- If value is `4`: a volume with the fuzzy false positive map between the model prediction and 
the ground truth is shown,
- If value is `5`: the model prediction and the ground truth on the same volume are shown.

The `--extra_file` argument allows to give a new configuration file to overwrite the
data, transform and model configuration files as well as the result directory. This aims
to allow evaluation on new data or transform without creating a whole new folder.