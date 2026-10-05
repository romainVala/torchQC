import numpy as np
import os
import commentjson
import random
import pandas as pd


def _get_lut(fname=None):
    """Get a FreeSurfer LUT."""

    dtype = [
        ("id", "<i8"),
        ("name", "U"),
        ("R", "<i8"),
        ("G", "<i8"),
        ("B", "<i8"),
        ("A", "<i8"),
    ]
    lut = {d[0]: list() for d in dtype}

    with open(fname) as fid:
        for line in fid:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            line = line.split()
            if len(line) != len(dtype):
                raise RuntimeError(f"LUT is improperly formatted: {fname}")
            for d, part in zip(dtype, line):
                lut[d[0]].append(part)
    lut = {d[0]: np.array(lut[d[0]], dtype=d[1]) for d in dtype}

    lut["name"] = [str(name) for name in lut["name"]]
    return lut


def convert_label_json_toctbl(json_file):
    with open(json_file, encoding="utf-8") as f:
        data = commentjson.load(f)

    if not ("labels" in data):
        raise ('error no labels keys in you json')
    output_path = json_file[:-5] + '.ctbl'
    label_dic = {k.replace(' ', '_'): int(v) for k, v in data['labels'].items()}

    print(label_dic)
    write_ctbl(label_dic, output_path)


def convert_label_csv_toctbl(csv_file,label_name,label_value):
    df = pd.read_csv(csv_file)

    if (not (label_name in df)) | (not (label_value in df)):
        raise ('error no labels keys in you json')
    output_path = csv_file[:-4] + '.ctbl'
    label_dic = {v1:v2  for v1,v2 in zip(df[label_name],df[label_value])}

    print(label_dic)
    write_ctbl(label_dic, output_path)


def get_freeColorLut(fref = '/data/romain/toolbox_python/romain/torchQC/script/FreeSurferColorLUT.txt'):

    lut = _get_lut(fref)
    names, ids = lut["name"], lut["id"]
    colors = np.array([lut["R"], lut["G"], lut["B"], lut["A"]], int).T
    atlas_ids = dict(zip(names, ids))
    colors = dict(zip(names, colors))
    return atlas_ids, colors



def write_ctbl(dicl, output_path, alpha=255, seed=None):
    if seed is not None:
        random.seed(seed)

    name_fs, col_fs = get_freeColorLut()

    with open(output_path, "w") as f:
        for label_name, label_id in sorted(dicl.items(), key=lambda x: x[1]):
            if label_id == 0:
                continue  # souvent on ignore le background
            if label_name == "GM" :
                r,g,b,a = col_fs['Left-Cerebral-Cortex']
            elif label_name == "WM" :
                r,g,b,a = col_fs['Left-Cerebral-White-Matter']
            elif label_name == "CSF" :
                r,g,b,a = col_fs['CSF']
            elif label_name == "Ventricles" :
                r,g,b,a = col_fs['3rd-Ventricle']
            elif label_name == "Cereb" :
                r,g,b,a = col_fs['Left-Cerebellum-Cortex']
            elif label_name == "Thal" :
                r,g,b,a = col_fs['Left-Thalamus']
            elif label_name == "Striatum" :
                r,g,b,a = col_fs['Left-Caudate']
            elif label_name == "RU" :
                r,g,b,a = col_fs['Red_nucleus_RN']
            elif label_name == "Dura" :
                r,g,b,a = col_fs['Dura']
            elif label_name == "vascular":
                r, g, b, a = col_fs['Artery']
            elif label_name == "Skull" :
                r,g,b,a = col_fs['Skull']
            elif label_name == "Head" :
                r,g,b,a = col_fs['Left-Eyeball']

            else:
                print(f'taking random color for label {label_name}')
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)

            f.write(f"{label_id} {label_name} {r} {g} {b} {alpha}\n")

def convert_FS_lut_to_ctbl(fname=None):
    lut = _get_lut(fname)
    output_path = fname[:-4] + '.ctbl'

    if os.path.isfile(output_path):
        print(f'Skipping because file {output_path} exist ')
        return

    alpha=255
    with open(output_path, "w") as f:
        for name,id,r,g,b in zip(lut['name'],lut['id'],lut['R'],lut['G'],lut['B']):
            f.write(f"{id} {name} {r} {g} {b} {alpha}\n")

    print(f'File {output_path} CREATED')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file to convert ', required=True, type=str)
    parser.add_argument('-t', '--type', help=' possible value FS | json  ', default='', required=False, type=str)
    parser.add_argument('-n', '--label_name', help=' name col for name in csv default Name', default='Name', required=False, type=str)
    parser.add_argument('-v', '--label_value', help=' name col for values in csv default Name', default='label', required=False, type=str)

    args = parser.parse_args()
    type = args.type
    fin = args.input
    if len(type)==0:
        if fin.endswith('.csv'):
            type = 'csv'
        elif fin.endswith('.json'):
            type = 'json'

    if args.type == "FS":
        convert_FS_lut_to_ctbl(fin)
    elif args.type == "json":
        convert_label_json_toctbl(fin)
    elif type == 'csv':
        convert_label_csv_toctbl(fin,args.label_name, args.label_value)
    else:
        print(f' unknow type {type} \n Choose either FS or json')

if __name__ == '__main__':
    main()
