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
    "folder": "/path_to_task_folder/",
    "data": "data.json",
    "transform": "transform.json",
    "loader": "loader.json",
    "model": "model.json",
    "train": "train.json",
    "visualization": "visualization.json"
}
```

#### data.json
This file tells the code where to find the data to use in the segmentation task and the repartition
between train, validation and test data. If `"shuffle"` is `true`, subjects are shuffled before
being assigned to train, validation and test sets. <br>
Either patterns using unix regular expressions or explicit paths can be used to list subjects.
Both can be used at the same time (patterns are read first).
```json
{
    "modalities": 
    {
        "t1": {"type": "intensity"},
        "label": {"type": "label"}
    },
    "patterns": 
    [
        {
            "root": "/path_to_t1/*/",
            "name_pattern": "([0-9]+)", 
            "modalities": {"t1": "/T1w/T1w_acpc_dc.nii.gz"}
        }, 
        {
            "root": "/path_to_label/suj_*/", 
            "name_pattern": "([0-9]+)", 
            "modalities": {"label": "grey.nii.gz"}
        }
    ], 
    "repartition": [0.7, 0.15, 0.15],
    "shuffle": true,
    "seed": 0
}
```

```json
{
    "modalities": 
    {
        "t1": {"type": "intensity"},
        "label": {"type": "label"}
    },
    "paths": 
    [
        {
            "name": "subject1",
            "modalities": 
            {
                "t1": "/path_to_t1_for_subject1/T1w/T1w_acpc_dc.nii.gz",
                "label": "/path_to_label_for_subject1/grey.nii.gz"
            }
        }, 
        {
            "name": "subject2",
            "modalities": 
            {
                "t1": "/path_to_t1_for_subject2/T1w/T1w_acpc_dc.nii.gz",
                "label": "/path_to_label_for_subject2/grey.nii.gz"
            }
        }
    ], 
    "repartition": [0.7, 0.15, 0.15],
    "shuffle": true,
    "seed": 0
}
```

#### transform.json
This file defines which transforms (preprocessing and data augmentation) are applied to train
and validation samples.
```json
{
    "train_transforms": 
    [
        {
            "name": "RandomBiasField",
            "attributes": {"seed":0}
        }, 
        {
            "name": "RandomNoise", 
            "attributes": {"seed":0}
        }, 
        {
            "name": "RandomFlip",
            "attributes": {"axes": [0], "seed": 0}
        },
        {
            "name": "OneOf", 
            "is_selection": "True",
            "transforms":
            [
                {
                    "proba": 0.8, 
                    "transform":
                    {
                        "name":"RandomAffine", 
                        "attributes": {"seed": 0}
                    }
                }, 
                {
                    "proba": 0.2,
                    "transform":
                    {
                        "name":"RandomElasticDeformation", 
                        "attributes": {"seed":0}
                    }
                }
            ]
        }
    ], 
    "val_transforms": []
}
```

#### loader.json
This file defines the behaviour of the `DataLoader`, using patches or not, the number of workers
used to load data (`-1` means all available workers are used) and the batch size.
```json
{
    "batch_size": 2,
    "num_workers": 4,
    "queue":
    {
        "attributes": 
        {
            "patch_size": 64, 
            "max_length": 64, 
            "samples_per_volume": 32
        }, 
    "sampler_class": "LabelSampler"
    }
}
```

#### model.json
This file defines the model to use, its parameters and if it is loaded from a saved model.
If attribute `"custom"` is `true`, it means that model was saved using `BaseNet` saving
method. 
```json
{
    "model": 
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
        }
    }
}
```

```json
{
    "model": 
    {
        "module": "unet",
        "name": "UNet3D", 
        "load": 
        {
            "custom": true, 
            "path": "/path_to_model"
        }
    }
}
```

#### train.json
This file defines
- the losses used to train the model (`"criteria"`),
- the optimizer (`"optimizer"`),
- the parameters of the logger to print and save logs (`"logger"`),
- where and how often the model and its performances are saved (`"save"`, the `"save_frequency"`
is in number of epochs while the `"record_frequency"` is in number of iterations),
- how often the model is evaluated on the validation set and which metrics are used to assess the 
quality of the segmentation on the validation set (`"validation"`, `"eval_frequency"` is in
number of iterations). The model is saved after every evaluation loop on the validation
set,
- how often inference on whole images is done, this is relevant only if patches are
 used during training (`"whole_image_inference_frequency"` in `"validation"`, this frequency is
 in number of epochs),
- the number of epochs.
```json
{
    "criteria": 
    [
        {
            "module":"segmentation.losses.dice_loss", 
            "name": "mean_dice_loss"
        }
    ], 
    "optimizer": 
    {
        "module": "torch.optim", 
        "name": "Adam", 
        "attributes": {"lr": 0.0001}
    }, 
    "logger": 
    {
        "log_frequency": 10, 
        "filename": "/path_to_save_logs/train_log.txt", 
        "name": "train_log"
    }, 
    "save": 
    {
        "save_model": true,
        "save_frequency": 1,
        "save_path": "/path_to_save_results/",
        "custom_save": true, 
        "record_frequency": 10
    }, 
    "validation": 
    {
        "whole_image_inference_frequency": 100, 
        "patch_size": 64, 
        "patch_overlap": 0, 
        "out_channels": 1, 
        "batch_size": 2, 
        "eval_frequency": 100, 
        "reporting_metrics": 
        [
            {
                "module":"segmentation.losses.dice_loss", 
                "name": "mean_dice_loss"
            }
        ]
    }, 
    "seed": 0, 
    "image_key_name": "t1", 
    "label_key_name": "label", 
    "n_epochs": 500
}
```

#### visualization.json
This file defines the visualization parameters. It uses `PlotDataset` to make plots.
```json
{
    "image_key_name": "t1",
    "label_key_name": "label",
    "subject_idx": [0, 1, 2, 3, 4], 
    "update_all_on_scroll": true
}
```

## Run the program
The main entry point of the program is the `segmentation_pipeline.py` file.
Therefore, in order to run the program, one can run the following command:
```shell script
python ./segmentation/segmentation_pipeline.py -f '/path_to_main_config_file/main.json'
```
The `--visualization` argument can be used to visualize samples, patches or both at the same 
time with the values 1, 2 or higher. Only samples and patches from the train set are used.