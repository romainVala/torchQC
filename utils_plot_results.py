import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from utils import remove_extension
from utils_file import get_parent_path, gfile, gdir
from segmentation.config import Config as cc
from termcolor import colored
import commentjson as json

def get_ep_iter_from_res_name(resname, nbit, batch_size=4):
    resname_no_ext = remove_extension(resname)
    ffn = [ff[ff.find('_ep') + 3:] for ff in resname_no_ext]
    key_list = []
    for fff, fffn in zip(ffn, resname):
        if '_it' in fff:
            ind = fff.find('_it')
            ep = int(fff[0:ind])
            it = int(fff[ind + 3:])*batch_size
            it = 4 if it==0 else it #hack to avoit 2 identical point (as val is done for it 0 and las of previous ep
        else:
            ep = int(fff)
            it = nbit
        key_list.append([fffn, ep, it])
    aa = np.array(sorted(key_list, key=lambda x: (x[1], x[2])))
    name_sorted, ep_sorted, it_sorted = aa[:, 0], aa[:, 1], aa[:, 2]
    ep_sorted = np.array([int(ee) for ee in ep_sorted])
    it_sorted = np.array([int(ee) for ee in it_sorted])
    ep_sorted = ep_sorted - ep_sorted[0] #so that the first is 0
    return name_sorted, ep_sorted, it_sorted

def my_read_csv_split_columns(fres):
    df = pd.read_csv(fres)

def convert_string_array_to_array(x):
    x = x.replace('[','')
    x = x.replace(']', '')
    x = ','.join(x.split())
    if x.startswith(','):
        x = x[1:]
    return np.array(eval(x))

def plot_train_val_results(dres, train_csv_regex='Train.*csv', val_csv_regex='Val.*csv',
                           prediction_column_name='prediction', target_column_name='targets',
                           target_scale=1, fign='Res', sresname=None):

    legend_str=[]
    col = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for ii, oneres in enumerate(dres):
        fresT = gfile(oneres, train_csv_regex)
        fresV=gfile(oneres, val_csv_regex)

        if len(fresV)==0:
            print('{} empty dir {} '.format(colored('Skiping','red'), get_parent_path(oneres)[1]))
            continue

        is_train = False if len(fresT)==0 else True
        if is_train: resT = [pd.read_csv(ff) for ff in fresT]

        resdir, resname = get_parent_path(fresV)
        nbite = len(resT[0]) if is_train else 80000
        fresV_sorted, b, c = get_ep_iter_from_res_name(resname, nbite)
        ite_tot = c+b*nbite
        ite_tottt = np.hstack([0, ite_tot])
        print(ite_tot)
        resV = [pd.read_csv(resdir[0] + '/' + ff) for ff in fresV_sorted]
        df_val = pd.concat(resV, ignore_index=True, sort=False)

        for rr in resV:
            if 'sample_time' not in rr:
                rr['sample_time'] = rr['batch_time'] / 4 #for old result always runed with batchsize 4
            if isinstance(rr[prediction_column_name][0], str):
                rr[prediction_column_name] = rr[prediction_column_name].apply(
                    lambda s: convert_string_array_to_array(s))
                rr[target_column_name] = rr[target_column_name].apply(
                    lambda s: convert_string_array_to_array(s))
        if is_train:
            for rr in resT:
                if 'sample_time' not in rr:
                    rr['sample_time'] = rr['batch_time'] / 4
                if isinstance(rr[prediction_column_name][0], str):
                    rr[prediction_column_name] = rr[prediction_column_name].apply(
                        lambda s: convert_string_array_to_array(s))
                    rr[target_column_name] = rr[target_column_name].apply(
                        lambda s: convert_string_array_to_array(s))

        if is_train:
            df_train = pd.concat(resT, ignore_index=True, sort=False)
            errorT = np.abs(df_train.loc[:,prediction_column_name].values -df_train.loc[:, target_column_name].values*target_scale)
            train_time = df_train.loc[:,'sample_time']
            #average between validation point itte_tot
            LmTrain = [np.mean(errorT[ite_tottt[ii]:ite_tottt[ii+1]]) for ii in range(0, len(ite_tot)) ]
            TimeTrain = [np.mean(train_time[ite_tottt[ii]:ite_tottt[ii + 1]]) for ii in range(0, len(ite_tot))]

        LmVal = [np.mean(np.abs(rr.loc[:,prediction_column_name]-rr.loc[:, target_column_name].values*target_scale)) for rr in resV]
        #LmVal = np.mean(np.abs(df_val.loc[:,prediction_column_name].values - df_val.loc[:,target_column_name].values*target_scale))
        TimeVal = [np.mean(rr.loc[:,'sample_time']) for rr in resV]

        plt.figure('MeanL1_'+fign); legend_str.append('V{}'.format(sresname[ii]));
        if is_train: legend_str.append('T{}'.format(sresname[ii]))
        plt.plot(ite_tot, LmVal,'--',color=col[ii])
        if is_train: plt.plot(ite_tot, LmTrain,color=col[ii], linewidth=6)

        plt.figure('Time_'+fign);
        plt.plot(ite_tot, TimeVal,'--',color=col[ii])
        if is_train: plt.plot(ite_tot, TimeTrain,color=col[ii], linewidth=6)

        #print some summary information on the results
        if not is_train:
            TimeTrain=0
        nb_res = len(resT) if is_train else 0
        np_iter = len(resT[0]) if is_train else 0

        totiter, mbtt, mbtv = ite_tot[-1] / 1000, np.nanmean(TimeTrain), np.mean(TimeVal)
        tot_time = nb_res * np_iter * mbtt + len(resV) * len(resV[0]) * mbtv
        percent_train = nb_res * np_iter * mbtt / tot_time
        tot_time_day = np.floor( tot_time/60/60/24 )
        tot_time_hour = (tot_time - tot_time_day*24*60*60) / 60/60
        print('Result : {} \t {} '.format(
            colored(get_parent_path(resdir[0])[1], 'green'), sresname[ii] ))
        print('\t{} epoch of {} vol {} val on {} vol Tot ({:.1f}%train) {} d {:.1f} h'.format(
            nb_res, np_iter, len(resV),len(resV[0]), percent_train, tot_time_day, tot_time_hour ))

        fj = gfile(oneres,'data.json')
        if len(fj)==1:
            data_struc = cc.read_json(fj[0])
            bs, nw = data_struc['batch_size'], data_struc['num_workers']
        else:
            bs, nw = 0, -1
        print('\tBatch size {} \tNum worker {} \t{:.1f} mille iter \t train/val meanTime {:.2f} / {:.2f} '.format\
                (bs, nw, totiter, mbtt, mbtv))


    plt.figure('MeanL1_'+fign);
    plt.legend(legend_str); plt.grid()
    ff=plt.gcf();ff.set_size_inches([15, 7]); #ff.tight_layout()
    plt.subplots_adjust(left=0.05, right=1, bottom=0.05, top=1, wspace=0, hspace=0)
    plt.ylabel('L1 loss')

    plt.figure('Time_'+fign);
    plt.legend(legend_str); plt.grid()
    plt.ylabel('time in second')

#df = pd.read_csv('/home/fabien.girka/Documents/segmentation/results/Train_ep1_it2.csv', index_col=0)
#rbf_hist = df['history_RandomBiasField']
#first_rbf_hist = json.loads(rbf_hist[0])

def get_pandadf_from_res_valOn_csv(dres, resname, csv_regex='res_valOn', data_name_list=None,
                                   select_last=None, target='ssim', target_scale=1):

    if len(dres) != len(resname) : raise('length problem between dres and resname')

    resdf_list = []
    for oneres, resn in zip(dres, resname):
        fres_valOn = gfile(oneres, csv_regex)
        print('Found {} <{}> for {} '.format(len(fres_valOn), csv_regex, resn))
        if len(fres_valOn) == 0:
            continue

        ftrain = gfile(oneres, 'res_train_ep01.csv')
        rrt = pd.read_csv(ftrain[0])
        nb_it = rrt.shape[0];

        resdir, resname_val = get_parent_path(fres_valOn)
        resname_sorted, b, c = get_ep_iter_from_res_name(resname_val, 0)

        if select_last is not None:
            if select_last<0:
                resname_sorted = resname_sorted[select_last:]
            else:
                nb_iter = b*nb_it+c
                resname_sorted = resname_sorted[np.argwhere(nb_iter > select_last)[1:8]]

        resV = [pd.read_csv(resdir[0] + '/' + ff) for ff in resname_sorted]
        resdf = pd.DataFrame()
        for ii, fres in enumerate(resname_sorted):
            iind = [i for i, s in enumerate(data_name_list) if s in fres]
            if len(iind) ==1: #!= 1: raise ("bad size do not find which sample")
                data_name = data_name_list[iind[0]]
            else:
                data_name = 'res_valds'

            iind = fres.find(data_name)
            ddn = remove_extension(fres[iind + len(data_name) + 1:])
            new_col_name = 'Mout_' + ddn
            iind = ddn.find('model_ep')
            if iind==0:
                transfo='raw'
            else:
                transfo = ddn[:iind - 1]

            if transfo[0] == '_': #if start with _ no legend ... !
                transfo = transfo[1:]

            model_name = ddn[iind:]
            aa, bb, cc = get_ep_iter_from_res_name([fres], nb_it)
            nb_iter = bb[0] * nb_it + cc[0]

            rr = resV[ii].copy()
            rr['evalOn'], rr['transfo'] = data_name, transfo
            rr['model_name'], rr['submodel_name'], rr['nb_iter'] = resn, model_name, str(nb_iter)
            rr[target] = rr[target] * target_scale
            resdf = pd.concat([resdf, rr], axis=0, sort=True)

        resdf['error'] = resdf[target] - resdf['model_out']
        resdf['error_abs'] = np.abs(resdf[target] - resdf['model_out'])
        resdf_list.append(resdf)

    return resdf_list


def plot_resdf(resdf_list, dir_fig=None,  target='ssim', split_distrib=True):

    for resdf in resdf_list :
        ee = np.unique(resdf.evalOn)
        resn = resdf['model_name'].values[0]
        zz = np.unique(resdf['model_name'])
        if len(zz)>1: raise('multiple model_name')

        if dir_fig is not None:
            dir_out_sub = dir_fig + '/' + resn +'/'
            if not os.path.isdir(dir_out_sub): os.mkdir(dir_out_sub)

        for eee in ee:
            dfsub = resdf.loc[resdf.evalOn == eee, :]
            #dfsub.transfo = dfsub.transfo.astype(str)
            fign = 'MOD_' + resn + '_ON_' + eee

            fig = plt.figure('Dist' + fign)
            #ax = sns.violinplot(x="transfo", y="error", hue="model_name", data=dfsub, palette="muted")
            ax = sns.violinplot(x="transfo", y="error", hue="transfo", data=dfsub, palette="muted")
            if split_distrib :
                nbline = int(dfsub.shape[0] / 2)
                plt.subplot(211);
                ax = sns.violinplot(x="nb_iter", y="error", hue="transfo", data=dfsub.iloc[:nbline, :], palette="muted")
                plt.grid()
                plt.subplot(212)
                ax = sns.violinplot(x="nb_iter", y="error", hue="transfo", data=dfsub.iloc[nbline:, :], palette="muted")
                plt.grid()
                ax.legend().set_visible(False);
                fig.set_size_inches([18, 6]); fig.tight_layout();   fig.suptitle(fign);

            else:
                ax = sns.violinplot(x="nb_iter", y="error", hue="transfo", data=dfsub, palette="muted")

            if dir_fig is not None:
                plt.savefig(dir_out_sub + 'Dist_' + fign + '.png');
                plt.close()

            g = sns.catplot(x="nb_iter", y="error_abs", hue="transfo", data=dfsub, palette="muted", kind="point",
                            dodge=True, legend_out=False)
            g.fig.suptitle('Error Abs' + fign)
            g.fig.set_size_inches([12, 5]);
            g.fig.tight_layout();
            if dir_fig is not None:
                plt.savefig(dir_out_sub + 'L1_' + fign + '.png');
                plt.close()

            sns.despine(offset=10, trim=True);

            g = sns.relplot(x=target, y="model_out", hue="nb_iter", data=dfsub,
                            palette=sns.color_palette("hls", dfsub.nb_iter.nunique()),
                            kind='scatter', col='transfo', col_wrap=3, alpha=0.5)
            axes = g.axes.flatten()
            for aa in axes:
                #aa.plot([0.5, 1], [0.5, 1], 'k')
                aa.plot([0.2, 2.2], [0.2, 2.2], 'k')
                plt.grid()

            g.fig.suptitle(fign, x=0.8, y=0.1)
            if dir_fig is not None:
                plt.savefig(dir_out_sub + 'Scat_' + fign + '.png');
                plt.close()

def transform_history_to_factor(r):
    name = 'TODO'
    if 'T_RandomAffine' in r :
        raff = r.T_RandomAffine
        if not isinstance(raff, float):
            par = json.loads(raff)
            name = 'Aff_S{:.1f}R{}'.format(par['scaling'][0], int(par['rotation'][0]))

    if 'T_RandomAffineFFT' in r :
        raff = r.T_RandomAffineFFT
        if not isinstance(raff, float):
            par = json.loads(raff)
            name = 'AffFFT_S{:.1f}R{}'.format(par['scaling'][0], int(par['rotation'][0]))

    if 'T_RandomElasticDeformation' in r :
        raff = r.T_RandomElasticDeformation
        if not isinstance(raff, float):
            name = 'Ela'

    if 'T_RandomBiasField' in r :
        raff = r.T_RandomBiasField
        if not isinstance(raff, float):
            name = 'Ela'

    return name


def parse_history(r ):
    def append_name_to_keys_in_dict(onedict, name_append):
        newdict = dict()
        for k, v in onedict.items():
            newdict[name_append + k] = v
        return newdict

    all_dict={}

    if 'T_RandomAffineFFT' in r :
        raff = r.T_RandomAffineFFT
        if isinstance(raff,float):
            row_dict =  {'scaling': [np.nan, np.nan, np.nan],
                         'rotation': [np.nan, np.nan, np.nan],
                         'oversampling': np.nan,
                         'noise_mean_T1w_1mm': np.nan,
                         'noise_std_T1w_1mm': np.nan,
                         'S_rot': np.nan,
                         'M_scale': np.nan}
        else:
            row_dict = json.loads(raff)
            row_dict['S_rot'] = np.sum(row_dict['rotation'])
            row_dict['M_scale'] = np.mean(row_dict['scaling'])

        row_dict = append_name_to_keys_in_dict(row_dict, 'A_FFT_')
        all_dict.update(row_dict)

    if 'T_RandomAffine' in r :
        raff = r.T_RandomAffine
        if isinstance(raff, float):
            row_dict = {'scaling': [np.nan, np.nan, np.nan],
                        'rotation': [np.nan, np.nan, np.nan],
                        'translation': np.nan,
                        'S_rot': np.nan,
                        'S_trans': np.nan,
                        'M_scale': np.nan}
        else:
            row_dict = json.loads(raff)
            row_dict['S_rot'] = np.sum(row_dict['rotation'])
            row_dict['S_trans'] = np.sum(row_dict['translation'])
            row_dict['M_scale'] = np.mean(row_dict['scaling'])

        row_dict = append_name_to_keys_in_dict(row_dict, 'Aff_')
        all_dict.update(row_dict)

    if 'T_RandomElasticDeformationSKIP' in r : #Skip becaus coarse_grid is too big and takes time to concatenate
        raff = r.T_RandomElasticDeformation
        if isinstance(raff, float):
            row_dict = {'coarse_grid': np.nan}
        else:
            row_dict = json.loads(raff)
            row_dict['coarse_grid'] = np.sum(row_dict['coarse_grid'])
        row_dict = append_name_to_keys_in_dict(row_dict, 'Ela_')
        all_dict.update(row_dict)

    if 'T_RandomMotionFromTimeCourse' in r :
        raff = r.T_RandomMotionFromTimeCourse
        if isinstance(raff, float):
            raise('Argg TODO but the dict  can change ... how to initial empty dict ?')
        else:
            row_dict = json.loads(raff)
        row_dict = append_name_to_keys_in_dict(row_dict, 'Mot_')
        all_dict.update(row_dict)

    return pd.Series(all_dict)
