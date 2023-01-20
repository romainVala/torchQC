
import torch,numpy as np,  torchio as tio
import json, os, seaborn as sns
import tqdm
import pandas as pd
from utils_file import get_parent_path, gfile, gdir

from utils_metrics import computes_all_metric, binarize_5D, get_results_dir

sns.set_style("darkgrid")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


labels_name = np.array(["bg","CSF","GM","WM","skin","vent","cereb","deepGM","bstem","hippo",])
selected_index = [1,2,3,5,6,7,8,9]
selected_label = torch.zeros(labels_name.shape); selected_label[selected_index]=1; selected_label = selected_label>0
labels_name = labels_name[selected_index]

mask_name, selected_label_mask = None, None
#mask_name = ["CSF", "WM"]; selected_label_mask = [1,3]

#labels_name=['GM'] ; selected_label = [1]
concat_CSF=True; do_coreg=True #la coregistration a ete fait sur la T1 (volume T1 ver volume T2) au lieu de le faire sur la GM
do_onehot=True; distance_metric=True; euler_metric=False; skip_if_exist=True
compute_by_patches = 'random'; save_patches=True; nb_patch_save=30; patch_size, batch_size = 32, 512;
compute_by_patches = 'grid'; save_patches=True; nb_patch_save=20; patch_size, batch_size = 16, 'grid';
compute_by_patches = 'no';  save_patches=False;

csv_file_name = 'metrics_coreg_affineGM.csv'
#csv_file_name = 'metrics.csv'  'metrics_eulerGM.csv'
#labels_name = ['GM']; selected_label = [2]
#csv_file_name = f'metrics_patch_{patch_size}_{batch_size}.csv'


resdir = '/data/romain/PVsynth/eval_cnn/baby/Article/'
#ress = gdir(resdir,'eval_T.*(mot|next)') ress = gdir(resdir,'eval_T.*(mot|_on5suj)')
ress = gdir(resdir,'eval_T.*(mot|_on5suj|OnT2)')
#ress = [ress[4]]
ress = ['/data/romain/PVsynth/eval_cnn/baby/Article/coreg/eval_T1_NOcoreg_model_5suj_motBigAff_ep30']
resname = get_parent_path(ress)[1]

resdir_label = None
#resdir_label = resdir + 'eval_T2_model_5suj_motBigAff_ep30'
#resdir_label = resdir + 'eval_T2_model_hcpT2_elanext_5suj_ep1'

# first construct dataFrame containing, label pred metric model_name
df_file = pd.DataFrame()
for nbres, resdir in enumerate(ress):
    print(f'reading result {resname[nbres]} ')
    #do_coreg = True if 'eval_T1' in resdir else False

    sujs = gdir(resdir,'.*')
    #sujs = [sujs[4]]

    for sujdir in sujs:
        #print(f'suj {sujdir} ')
        sujname = get_parent_path(sujdir)[1]
        if resdir_label is None:
            label_file = gfile(sujdir, 'label.*nii', opts={"items": 1})
        else:
            label_dir = gfile(resdir_label, sujname)
            label_file = gfile(label_dir, 'pred.*nii', opts={"items": 1})

        pred_file = gfile(sujdir, 'pred.*nii', opts={"items": 1})
        input_file = gfile(sujdir, 'data.nii', opts={"items": 1})
        csv_file = sujdir + '/' + csv_file_name
        res_dict = {'sujname': sujname, 'model_name': resname[nbres],'finput':input_file,  'flabel': label_file, 'fpred': pred_file, 'csv_file': csv_file}
        df_file = pd.concat([df_file, pd.DataFrame([res_dict])], ignore_index=True)


#then loop over rows and compute metrics
df = pd.DataFrame()
for onsuj in df_file.itertuples():
    sujname = onsuj.sujname
    resname = onsuj.model_name
    label_file = onsuj.flabel;    pred_file = onsuj.fpred; input_file = onsuj.finput
    csv_file = onsuj.csv_file

    print(f'suj {sujname} model  {resname} ')

    if os.path.exists(csv_file) & skip_if_exist:
        print(f'SKIP file {csv_file} exist')
    else:
        l_img = tio.LabelMap(label_file); p_img = tio.LabelMap(pred_file); d_img = tio.ScalarImage(input_file)

        if do_onehot:
            thot = tio.OneHot();            l_img = thot(l_img);
            #force total number of classe, if missing class in predictions
            thot = tio.OneHot(num_classes=l_img.data.shape[0]);  p_img = thot(p_img)


        if concat_CSF:
            remapping = {"0":0, "1":1, "2":2,"3":3,"4":4, "5": 1, "6":5, "7":6, "8":7, "9":8}
            tremap = tio.RemapLabels(remapping)
            l_img = tremap(l_img); p_img = tremap(p_img)
            labels_name = np.array(["bg", "CSF", "GM", "WM", "skin", "cereb", "deepGM", "bstem", "hippo", ])
            selected_index = [1, 2, 3, 5, 6, 7, 8]
            selected_label = torch.zeros(labels_name.shape);            selected_label[selected_index] = 1;            selected_label = selected_label > 0
            labels_name = labels_name[selected_index]


        if compute_by_patches == 'random':
            subject = tio.Subject({'lab': l_img, 'pred': p_img, 'data': d_img})
            tio_ds = tio.SubjectsDataset([subject], transform=None)
            tio_samp = tio.LabelSampler(patch_size=patch_size, label_name='lab', label_probabilities={2 : 1})
            train_queue = tio.Queue(tio_ds, max_length = batch_size, samples_per_volume=batch_size, sampler=tio_samp)
            labels_name=['GM'] ; selected_label = [2]

            train_loader = torch.utils.data.DataLoader(train_queue, batch_size=batch_size ) #, collate_fn=struct['collate_fn'])

            df_one = pd.DataFrame()
            for ii, data in enumerate(train_loader):
                print(f'{ii}')
                for nb_batch, (pred, target, inputdata, loc) in enumerate(zip(data['pred']['data'], data['lab']['data'],data['data']['data'],  data['location'])):
                    cc = computes_all_metric(pred.unsqueeze(0), target.unsqueeze(0), labels_name, indata=inputdata.unsqueeze(0),
                                         selected_label=selected_label, distance_metric=distance_metric, euler_metric=euler_metric)
                    res_dict = {'sujname': sujname, 'model_name': resname, 'finput': input_file, 'flabel': label_file, 'fpred': pred_file, 'location' : loc.numpy()}
                    res_dict.update(cc)
                    df_one = pd.concat([df_one, pd.DataFrame([res_dict]) ])

        elif compute_by_patches == 'grid':
            subject = tio.Subject({'lab': l_img, 'pred': p_img, 'data': d_img})
            labels_name=['GM'] ; selected_label = [2]

            # patchees from maxpooling
            GM = subject.lab.data[selected_label] #(subject.lab.data == 2).float()
            orig_shape = GM.shape[1:]
            #for poolingn unpooling, it should be a multiple of patch_size
            #treshape = tio.EnsureShapeMultiple(patch_size)
            #GM = treshape(GM)

            maxpool = torch.nn.MaxPool3d(kernel_size=patch_size, return_indices=True)
            avgpool = torch.nn.AvgPool3d(kernel_size=patch_size)
            (gm_pool, index) = maxpool(GM.float())
            gm_avg = avgpool(GM.float())

            unmaxpool = torch.nn.MaxUnpool3d(patch_size)
            #gm_unpool = unmaxpool(gm_pool, index, output_size=GM.shape[1:])
            gm_unpoolavg = unmaxpool(gm_avg, index, output_size=GM.shape[1:])
            #tpad = tio.CropOrPad(target_shape=subject.lab.shape[1:])
            #gm_unpool = tpad(gm_unpool)
            #subject.add_image(tio.ScalarImage(tensor=gm_unpool, affine=subject.lab.affine), 'gm')
            subject.add_image(tio.ScalarImage(tensor=gm_unpoolavg, affine=subject.lab.affine), 'gmavg')

            gs = tio.GridSampler(subject, patch_size)
            #select_patch = []
            df_one = pd.DataFrame()
            found_patch = 0
            for suj_patch in gs:
                if suj_patch.gmavg.data.sum() > 0.05: #select only patch that contain more than 10% of GM (from the label) 204 vox for a 16**3 patch
                    found_patch+=1
                    #print(f'found_patch val {suj_patch.gm.data.sum() } Avg {suj_patch.gmavg.data.sum() }  ' )
                    #select_patch.append([suj_patch.location])
                    pred = suj_patch.pred.data.unsqueeze(0)
                    target = suj_patch.lab.data.unsqueeze(0)
                    inputdata = suj_patch.data.data.unsqueeze(0)
                    location = suj_patch.location.numpy()
                    nb_pts_label, nb_pts_pred = target[0,selected_label,...].sum().numpy(),  pred[0,selected_label,...].sum().numpy()
                    res_dict = {'sujname': sujname, 'model_name': resname, 'finput': input_file, 'flabel': label_file,
                                'fpred': pred_file, 'location': location,
                                'nb_pts_label': nb_pts_label, 'nb_pts_pred': nb_pts_pred}

                    if nb_pts_pred*nb_pts_label > 0:
                        cc = computes_all_metric(pred, target, labels_name,indata=inputdata,selected_lab_mask=selected_label_mask,
                                                 lab_mask_name=mask_name,selected_label=selected_label,
                                                 distance_metric=distance_metric, euler_metric=euler_metric)
                        res_dict.update(cc)
                    df_one = pd.concat([df_one, pd.DataFrame([res_dict])])
        else:
            res_dict = {'sujname': sujname, 'model_name':  resname, 'label': label_file, 'pred': pred_file }
            res_dict.update(computes_all_metric(p_img.data.unsqueeze(0), l_img.data.unsqueeze(0), labels_name, selected_label=selected_label,
                                                distance_metric=distance_metric, euler_metric=euler_metric))
            df_one = pd.DataFrame([res_dict])

        if (save_patches & (compute_by_patches is not None) ) :

            patch_select_all = torch.zeros([1] + list(l_img.data.shape[1:]))
            locations = df_one['location'].values
            for location in locations:
                patch_one = torch.zeros([1] + list(l_img.data.shape[1:]))
                patch_one[0, location[0]:location[3], location[1]:location[4], location[2]:location[5]] = 1;
                patch_select_all += patch_one
            patch_img = tio.ScalarImage(tensor=patch_select_all, affine=l_img.affine)
            patch_img.save(os.path.dirname(l_img.path[0]) + f'/patch_all_{csv_file_name[:-4]}.nii.gz')

            #write patch max Sdis_GM dice_GM
            dfsort = df_one.sort_values('dice_GM', ascending=False)
            locations = dfsort['location'].values
            patch_select = torch.zeros([nb_patch_save] + list(l_img.data.shape[1:]))
            for nbp in range(nb_patch_save):
                location  = locations[nbp]
                patch_select[nbp,location[0]:location[3],location[1]:location[4],location[2]:location[5]] = dfsort['dice_GM'].values[nbp];
            patch_img = tio.ScalarImage(tensor=patch_select, affine=l_img.affine)
            patch_img.save(f'{os.path.dirname(l_img.path[0])}/patch_dice_GM_{csv_file_name[:-4]}.nii.gz')

            dfsort = df_one.sort_values('Sdis_GM', ascending=False)
            locations = dfsort['location'].values
            patch_select = torch.zeros([nb_patch_save] + list(l_img.data.shape[1:]))
            for nbp in range(nb_patch_save):
                location  = locations[nbp]
                patch_select[nbp,location[0]:location[3],location[1]:location[4],location[2]:location[5]] = dfsort['Sdis_GM'].values[nbp];
            patch_img = tio.ScalarImage(tensor=patch_select, affine=l_img.affine)
            patch_img.save(f'{os.path.dirname(l_img.path[0])}/patch_Sdis_GM_{csv_file_name[:-4]}.nii.gz')

        if do_coreg:
            print(f' dice after coregistration ')
            suj = tio.Subject(dict(lab=l_img, pred=p_img))
            suj.add_image(tio.ScalarImage(tensor=suj.pred.data[2].unsqueeze(0), affine=suj.pred.affine), 'predGM')
            suj.add_image(tio.ScalarImage(tensor=suj.lab.data[2].unsqueeze(0), affine=suj.pred.affine), 'labGM')
            tcoreg = tio.Coregister(target='predGM', default_parameter_map='affine',estimation_mapping={'labGM': ['lab']})
            sujcoreg = tcoreg(suj)

            lab_bin = binarize_5D(sujcoreg.lab.data.unsqueeze(0), add_extra_to_class=0)
            sujcoreg['lab']['data'] = lab_bin[0]
            res_dict = {'sujname': sujname, 'model_name': f'coreg_{resname}', 'label': label_file, 'pred': pred_file}

            dice = computes_all_metric(suj.pred.data.unsqueeze(0), sujcoreg.lab.data.unsqueeze(0), labels_name,
                                       selected_label=selected_label, distance_metric=distance_metric, euler_metric=euler_metric)
            #dicec = {f'coreg_{k}': v for k,v in dice.items() }
            res_dict.update(dice)
            df_one = pd.concat([df_one,  pd.DataFrame([res_dict]) ], ignore_index=True)
        df_one.to_csv(csv_file)

        df = pd.concat([df, df_one], ignore_index=True)

#df.to_csv('all_metric_res.csv')
