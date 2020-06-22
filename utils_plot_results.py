import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import remove_extension
from utils_file import get_parent_path, gfile, gdir


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
    return name_sorted, ep_sorted, it_sorted


def plot_train_val_results(dres, train_csv_regex='Train.*csv', val_csv_regex='Val.*csv',
                           prediction_column_name='prediction', target_column_name='targets',
                           target_scale=1, fign='Res', sresname=None):

    legend_str=[]
    col = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for ii, oneres in enumerate(dres):
        fresT = gfile(oneres, train_csv_regex) #gfile(oneres,'res_train.*csv')
        fresV=gfile(oneres, val_csv_regex)

        is_train = False if len(fresT)==0 else True
        if is_train: resT = [pd.read_csv(ff) for ff in fresT]

        resdir, resname = get_parent_path(fresV)
        nbite = len(resT[0]) if is_train else 80000
        fresV_sorted, b, c = get_ep_iter_from_res_name(resname, nbite)
        resV = [pd.read_csv(resdir[0] + '/' + ff) for ff in fresV_sorted]

        ite_tot = c+b*nbite
        if is_train:
            #errorT = np.hstack( [ np.abs(rr.model_out.values - rr.loc[:,target].values*target_scale) for rr in resT] )
            errorT = np.hstack( [ np.abs(rr.loc[:,prediction_column_name].values -
                rr.loc[:, target_column_name].values*target_scale) for rr in resT] )
            ite_tottt = np.hstack([0, ite_tot])
            LmTrain = [ np.mean(errorT[ite_tottt[ii]:ite_tottt[ii+1]]) for ii in range(0,len(ite_tot)) ]

        #LmVal = [np.mean(np.abs(rr.model_out-rr.loc[:,target].values*target_scale)) for rr in resV]
        LmVal = [np.mean(np.abs(rr.loc[:,prediction_column_name].values -\
                                rr.loc[:,target_column_name].values*target_scale)) for rr in resV]
        plt.figure('MeanL1_'+fign); legend_str.append('V{}'.format(sresname[ii]));
        if is_train: legend_str.append('T{}'.format(sresname[ii]))
        plt.plot(ite_tot, LmVal,'--',color=col[ii])
        if is_train: plt.plot(ite_tot, LmTrain,color=col[ii], linewidth=6)

    plt.legend(legend_str); plt.grid()
    ff=plt.gcf();ff.set_size_inches([15, 7]); #ff.tight_layout()
    plt.subplots_adjust(left=0.05, right=1, bottom=0.05, top=1, wspace=0, hspace=0)


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
