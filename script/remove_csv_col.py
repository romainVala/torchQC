import pandas as pd, glob, os
from utils_file import get_all_recursif_dir

col_to_removes = ["T_ElasticDeformation", "T_MotionFromTimeCourse","T_BiasField"]

d = get_all_recursif_dir(os.getcwd())

for dirs in d:
    fcsv = glob.glob(dirs+'/T*csv')
    if len(fcsv):
        print(dirs)

    for ff in fcsv:
        try:
            df = pd.read_csv(ff)
        except:
            print(f'FILE CORRUPT {ff}')
            continue

        save=False; removed_key=[]
        for col_to_remove in col_to_removes:
            if col_to_remove in df:
                removed_key.append(col_to_remove)
                print(f'removing {col_to_remove} ')
                df.drop(col_to_remove, axis=1, inplace=True)
                save=True
        if save:
            df.to_csv(ff)
            print(f' removing {removed_key}    saving new file {ff}')

