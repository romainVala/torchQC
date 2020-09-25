# Segmentation

This package aims to provide a generic code based on configuration files
to train a Pytorch neural network on a 3D segmentation task.

## Configuration files
A few configuration files are needed to run the code and should be referenced
in the `main.json` file (the name of the file is not important).

#### main.json
This file references all the configuration files needed to run the code.
```json5
{
    "data": "some_path/data.json",                      // Path to data config file
    "transform": "some_path/transform.json",            // Path to transform config file
    "model": "some_path/model.json",                    // Path to model config file
    "run": "some_path/run.json"                         // Path to run config file
    // "visualization": "some_path/visualization.json"  // Path to visualization config file
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
```json5
{
    "images":   // The different TorchIO images present in each subject
    {
        "t1": {     // Name of the image
            "type": "intensity",    // Type of the image (from TorchIO)
            "components":  ["t1"],  // Components of the image, i.e. the different 3D images composing the 4D image
            "attributes": {}        // Attributes of the image 
        },
        "label": {"type":  "label", "components":  ["GM", "CSF", "WM"]}
    },
    "patterns": // The patterns used to retrieve the data
    [
        {
            "root": "/path_to_t1/*/",   // The root of the pattern, used as a regex to find multiple files/folders
            "name_pattern": "([0-9]+)", // A regex to get the name of the subject from the found files/folders
            "components": {             // The components to be found with this pattern
                "t1": {                     // Name of the component
                    "path": "T1w/T1w_acpc_dc.nii.gz",   // Additional path to the component
                    "image": "t1"                       // Name of the image to which add the component
                }
            }
            // "list_name": "train",    // Name of the set to add the subject to (one of "train", "val" or "test")
            // "prefix": "",            // Prefix to add to the subject name generated from "name_pattern"
            // "suffix": ""             // Suffix to add to the subject name generated from "name_pattern"
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
    "repartition": [0.7, 0.15, 0.15], // The repartition between train, validation and test sets
    "subject_shuffle": true,          // If subjects are shuffled before being split between sets
    "subject_seed": 0,                // Shuffle subject seed
    "image_key_name": "t1",           // Name of the subject key used as input for the model
    "label_key_name": "label",        // Name of the subject key used as target for the model
    "labels": ["GM", "CSF", "WM"],    // List of component names in the target to create a mapping between channels and labels
    "batch_size": 2,                  // Batch size of the DataLoader
    "num_workers": 0,                 // Number of workers used to load data in the DataLoader
    "queue":                          // Queue to load patches from different subjects and give them to the model
    {
        "attributes":                     // Attributes of the queue
        {
            "max_length": 8,                  // Maximum number of patches in the queue
            "samples_per_volume": 4           // Number of patches sampled per subject
        }, 
        "sampler":                        // The sampler used to extract patches
        {
            "name": "UniformSampler",         // Name of the sampler class
            "module": "torchio.data.sampler", // Module that contains the sampler class
            "attributes":                     // Attributes of the sampler
            {
                "patch_size": 64                  // Patch size
            }
        }
    },
    "collate_fn":                     // Function to collate elements in batch
    {
        "name": "history_collate",                  // Name of the collate function
        "module": "segmentation.collate_functions"  // Module that contains the collate function
    },
    "batch_shuffle": true,            // If batches are shuffled
    "batch_seed": 0                   // Shuffle batch seed
    // "csv_file": []                 //
    // "load_sample_from_dir": []     // 
}
```

```json5
{
    "images": 
    {
        "t1": {"type": "intensity", "components": ["t1"]},
        "label": {"type": "label", "components": ["GM", "CSF", "WM"]}
    },
    "paths":  // The paths to the different subjects
    [
        {
            "name": "subject1",   // The name of the subject
            "components":         // The components present the subject
            {
                "t1": {               // The name of the component
                    "path": "/path_to_t1_for_subject1/T1w/T1w_acpc_dc.nii.gz",  // Path to the component
                    "image": "t1"                                               // Name of the image to which add the component
                },
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
```json5
{
    "train_transforms":   // List of transforms applied to the training set
    [
        {
            "is_custom": false,         // If transform does not come from TorchIO
            "name": "RandomBiasField",  // Name of the transform
            // "module": "some.module", // Name of the module containing the transform if custom is true
            "attributes": {             // Attributes of the transform
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
            "is_selection": "True", // If the transform selects or composes with a list of transforms
            "transforms":           // The list of transforms the transform acts on
            [
                {
                    "prob": 0.8,        // The probability that this transform is selected
                    "transform":        // The transform to select
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
    "val_transforms": [],   // List of transforms applied to the training set
    "post_transforms": [],  // Transforms to apply on the data generated by the model
}
```
`"train_transforms"` and `"val_transforms"` are mandatory.

#### model.json
This file defines the model to use, its parameters and if it is loaded from a saved model.
If attribute `"last_one"` is `true`, it means that last saved model will be used. If no model
has been saved so far, a new model will be used. To use a model saved at a specific path,
`"last_one"` must be set to `false` and `"path"` must be filled with the path to the model.
```json5
{
    "module": "unet",   // Name of the module containing the model class
    "name": "UNet3D",   // Name of the model class
    "attributes":       // Attributes of the model
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
    "last_one": true,       // If the last saved model in the result directory is used
    "device": "cuda"        // Device the tensors are sent to
    // "input_shape": null  // Input shape seen by the model, only used to print model summary
    // "eval_csv_basename": null
}
```

```json5
{
    "module": "unet",
    "name": "UNet3D", 
    "last_one": false,
    "path": "/path_to_model"  // Path to the saved model if last_one is false
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
```json5
{
    "activation": { // Model activation function
        "attributes": {                   // Attributes of the activation function
            "dim": 1
        },
        "module": "torch.nn.functional",  // Name of the module containing the activation function
        "name": "softmax"                 // Name of the activation function
    },
    "criteria":   // Losses used for training, the loss is the weighted sum of the different losses
    [
        {
            "module": "segmentation.losses.dice_loss",  // Name of the module containing the loss
            "name": "Dice",                             // Name of the loss class
            "method": "mean_dice_loss",                 // Method of the loss class
            "weight": 1,                                // Weight of the loss
            "channels": null,                           // List of channels used to compute the loss, if null, all channels are used
            "mask": null,                               // Channel from the target used as a mask to compute the loss
            "mask_cut": [0.99, 1],                      // Thresholds used for the mask
            "binarize_target": false,                   // If target should be binarized
            "binarize_prediction": false,               // If prediction should be binarized
            "binary_volumes": false                     // If prediction and target are given as binary label maps
            // "activation": {...}                      // Activation function, default uses the model activation function
        }
    ], 
    "optimizer":  // Optimizer used by the network
    {
        "module": "torch.optim",      // Name of the module containing the optimizer
        "name": "Adam",               // Name of the optimizer class
        "attributes": {"lr": 0.0001}, // Attributes of the optimizer
        "lr_scheduler":               // Learning rate scheduler
        {
            "module": "torch.optim.lr_scheduler", // Name of the module containing the scheduler
            "name": "ReduceLROnPlateau",          // Name of the scheduler class
            "attributes":                         // Attributes of the scheduler
            {
                "mode": "min",
                "patience": 6
            }
        }
    }, 
    "save":     // Information related to saving results
    {
        "record_frequency": 10,                         // Frequency (in terms of iterations) at which CSV files are saved
        "batch_recorder": "record_segmentation_batch",  // RunModel method used to record CSV files
        "prediction_saver": "save_volume",              // RunModel method used to save model outputs
        "label_saver": "save_volume",                   // RunModel method used to save labels (may be useful if transforms were applied)
        "save_bin": false,                              // If binary versions of the model outputs should be saved
        "save_channels": null,                          // List of channel names to be saved, mapping is done using "labels" from data.json
        "save_threshold": 0,                            // Threshold under which values are set to 0 before saving (reduce file size)
        "save_volume_name": "prediction",               // Name of the saved file
        "split_channels": false                         // If 4D image should be saved as multiple 3D images
    }, 
    "validation":   // Information related to evaluation
    {
        "whole_image_inference_frequency": 100,         // Frequency (in number of epochs) at which inference on the whole volume is made (relevant if patches)
        "patch_overlap": 8,                             // Overlap between patches when inferring the whole volume (relevant if patches)
        "eval_frequency": 100,                          // Validation frequency (in number of iterations)
        "prefix_eval_results_dir": null,                // Prefix to the result directory to which save predictions and evaluation CSV file, if null, default result directory is used
        "save_predictions": false,                      // If predictions should be saved during evaluation
        "save_labels": false,                           // If labels should be saved during evaluation
        "reporting_metrics":                            // Reported metrics during evaluation, work the same way as criteria
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
                "reported_name": "FP_grey_in_white",    // The name appearing in the CSV prefixed by "metric_"
                "channels": ["grey"]
            }
        ]
    }, 
    "seed": 0,                              // PyTorch seed used at the beginning of training 
    "log_frequency": 10,                    // Logging frequency (in number of iterations)
    "n_epochs": 500,                        // Number of training epochs
    "data_getter": "get_segmentation_data", // RunModel method used to retrieve input data and target
    "apex_opt_level": null                  // Apex optimization level, if null, Apex is not used, available values are "O0", "O1", "O2" and "O3"
}
```
`"criteria"`, `"optimizer"`, `"save"`, `"validation"` and `"n_epochs"` are mandatory.

In the `"optimizer"` dictionary, `"name"` and `"module"` are mandatory. <br>
If `"lr_scheduler"` is present in the `"optimizer"` dictionary, it must 
have `"name"` and `"module"`.
In the `"save"` dictionary, `"record_frequency"` is mandatory. <br>

Available methods from RunModel to retrieve input data and target using `"data_getter"` are 
`"get_segmentation_data"`, `"get_regression_data"`, `"get_regress_motion_data"` and 
`"get_regress_random_noise_data"`.
Available methods from RunModel to record data from a batch using `"batch_recorder"`
are `"record_segmentation_batch"` and `"record_regression_batch"`.
Only `"save_volume"` is available to save predictions or labels using 
`"prediction_saver"` or `"label_saver"`.

The optimizer is only used at training time so a model with no parameters like 
the Identity can be used for evaluation.

#### visualization.json
This file defines the visualization parameters. It uses `PlotDataset` to make plots.
```json5
{
    "kwargs":     // Attributes of the PlotDataset class, "image_key_name" and "label_key_name" are taken from data.json
    {
        "subject_idx": [0, 1, 2, 3, 4], 
        "update_all_on_scroll": true,
        "nb_patches": 4
    },
    "set": "val"  // Set to plot, either "train", "val" or "test"
}
```
When plotting volumes, you can navigate through slices by scrolling on the volumes,
if `"update_all_on_scroll"` is `true`, all similar views will be updated.

When plotting labels with several channels, you can navigate channels with up
and down keys.

When plotting subjects on more than one figure, you can navigate through figures
using pageup or pagedown keys.

#### extra_file.json
This file can overwrites `data.json`, `transform.json`, `model.json`, `run.json`
and/or the results directory.
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

#### grid_search.json
This file can create a set of experiments by overwriting some values from
the precedent files. The cartesian product of all the overwritten values is 
used to generate the new experiments. A `"prefix"` is used to determine 
where to save the new experiments, to this is added a string per experiment,
generated directly from the values or explicitly specified. <br>
The example in the json would create 4 experiments with `"results_dir"` being
`"result_root1_model1"`, `"result_root1_model2"`, `"result_root2_model1"` and
`"result_root2_model2"`. 
```json5
{
    "data.patterns.1.root": {   // The key to override, it targets the "root" attribute of the first element of "patterns" from data.json
        "prefix": "result",         // The prefix of the new result directories
        "values": [                 // The new values for the targeted key
            "/root1/*.nii.gz",
            "/root2/*.nii.gz"
        ],
        "names": [                  // The names to generate the new result directories from
            "root1",
            "root2"
        ]
    },
    "model.path": {             // "path" attribute from model.json is targeted
        "prefix": "model",
        "values": [
            "model1.pth.tar",
            "model2.pth.tar"
        ],
        "names": [
            "model1",
            "model2"
        ]
    }
}
```

#### create_jobs.json
This file defines the parameters used to create jobs if user chooses to create
jobs instead of directly running the experiments.
```json5
{
    "job_name": "eval_model_jobs",  // Name of the job
    "output_directory": "jobs"      // Jobs' output directory
    // "cluster_queue": "bigmem,normal",
    // "cpus_per_task": 1,
    // "mem": 4000,
    // "walltime": "12:00:00",
    // "job_pack": 1
}
```

## Run the program
The main entry point of the program is the `segmentation_pipeline.py` file.
Therefore, in order to run the program, one can run the following command:
```shell script
python ./segmentation/segmentation_pipeline.py -f '/path_to_main_config_file/main.json' -r 'results'
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

The `--create_jobs_file` argument allows to provide a new configuration file
to specify options on how to create jobs. If a file is given the experiment 
will not be run but jobs will be created instead.

The `--grid_search_file` argument allows to give a grid search configuration file
to overwrite specific keys from the previous configuration files and run
new experiments with specific result directories.

The `--max_subjects_per_job` argument allows to split a job in several ones
to limit the number of subjects a job will handle. Has no effect if
`--create_jobs_file` is empty and should only be used for evaluation or inference.