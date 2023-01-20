import pandas as pd
import glob

col_to_removes = ["T_ElasticDeformation", "T_MotionFromTimeCourse"]

fcsv = glob.glob('*csv')

for ff in fcsv:

    df = pd.read_csv(ff)
    save=False
    for col_to_remove in col_to_removes:
        if col_to_remove in df:
            print(f'removing {col_to_remove} saving new file {ff}')
            df.drop(col_to_remove, axis=1, inplace=True)
            save=True
    if save:
        df.to_csv(ff)
        print(f'     saving new file {ff}')

