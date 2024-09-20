import pandas as pd, glob, os
from utils_file import get_all_recursif_dir, gfile
from natsort import natsorted

def delete_file_list(ff):
    for file in ff:
        os.remove(file)

d = get_all_recursif_dir(os.getcwd())

for dirs in d:
    f = gfile(dirs, 'opt_ep.*tar')
    delete_file_list(f)
    f = gfile(dirs, 'amp_ep.*tar')
    delete_file_list(f)

    f = gfile(dirs, 'model_ep')
    f = natsorted(f)
    if len(f)==0:
        print(f'empty dir {dirs}')
        continue

    f.pop(-1) #always keep last one
    print(f'working in {dirs}')

    for ff in f:
        fname = os.path.basename(ff)
        res = [i for i in range(len(fname)) if fname.startswith('_', i)]
        if len(res)<=2:
            continue
        fname[res[1]]
        number = int(fname[res[1]-1])
        if number!=0:
            print(f'deleting {ff}')
            os.remove(ff)
        #else:
        #    print(f'keeping {fname}')
