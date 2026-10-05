import pandas as pd

from utils_file import gfile, gdir, get_parent_path, addprefixtofilenames, r_move_file,delete_file_list, r_mkdir
import torchio as tio
from script.export_nnunet import nnunet_siam_pred_job, make_validation_csv,make_validation_csv_HCP_repeate,make_validation_csv_HCP_autoref
from utils_job_preproc import reg_compose2, reg_applyw, reg_aff, job_mask_mrt
from script.export_nnunet import create_SynthSeg_job, create_AssN_job, create_FS_job
from utils_labels import remap_filelist, get_fastsurfer_remap, get_remapping, create_mask

rd='/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set'
din = [] #all testset
din += gdir(rd,['Synth_Atro_Ctx','vol'])
din += gdir(rd,['dHcp_old075','vol_T1'])
din += gdir(rd,['MindBoggle101','vol_T1'])
din += gdir(rd,['(MICCAIstd_testset_suj20|ultracortex|DBB)','vol_T'])
din += gdir(rd,['(HCP_test)','(vol_.*07$|realign$)'])
din += gdir(rd,['ULTRA_all','vol_ct_clip'])
nnunet_siam_pred_job(din,model_num_list=[8,12,13,14,15,],res_str='res')

models=['FastSurfer','SynthSeg','SuperSynth','gouhfi','715_NO','708','712_NO','714','713','610']
num_models=['FastSurfer','SynthSeg','SuperSynth','gouhfi','15','8','12','14','13','610']

models=['715_NO','708','712_NO','714','713','610']
num_models=['15','8','12','14','13','610']
fcsvs = gfile(din,'csv'); outdir_csv =  rd+'/csv_validat/'
deval = gdir(outdir_csv,'results$');#deval=gdir(get_parent_path(din)[0],['eval','results$'])
deval_name = get_parent_path(deval,3)[1]
metrics = "dice volume hausdorff"
#test eval exist
for fcsv in fcsvs:
#for fcsv in fcsvs:
    #print(f"\t {dname}")
    for model_name,num_model in zip(models, num_models):
        fres = []; #
        #fres = gfile(dd,model_name)
        if len(fres)==0:
            print(f'missing model {model_name} for {get_parent_path(fcsv)[1]}')
            if num_model=='8':
                make_validation_csv([fcsv], outdir_csv, pred_regex=model_name,
                                    auto_DS='^DS708', metrics_name=metrics)
            elif num_model == 'FastSurferNNNNNNN':
                make_validation_csv([fcsv], outdir_csv, pred_regex=model_name,
                                    auto_DS='^DSFree_lesion_remap', metrics_name=metrics)
                                    # Mindboggle has label 17 ? auto_DS='^DSFree_remap', metrics_name=metrics)
            elif ('Synth' in num_model) or ('gouhfi' in num_model) or ('FastSur' in num_model):
                if 'DBB' in fcsv:
                    make_validation_csv([fcsv], outdir_csv, pred_regex=model_name,
                                        auto_DS='^DSFree_remap', metrics_name=metrics)

                else:
                    make_validation_csv([fcsv], outdir_csv, pred_regex=model_name,
                                        auto_DS='^DSFree_lesion_remap', metrics_name=metrics)

            else :
                make_validation_csv([fcsv], outdir_csv, pred_regex=model_name,
                                auto_DS='^DS712',metrics_name= metrics)
        else :
            print(f"\t OK skipping {model_name}")

#test prediction exist
for dd in din:
    for model_name,num_model in zip(models, num_models):
        dres = gdir(dd,model_name)
        print(f"found {len(dres)} for {model_name} in {dd}")

        if len(dres)>1:
            print(f"found {len(dres)} for {model_name} in {dd}")

        if len(dres)==0:
            #print(f"missing {model_name} in {dd}")
            print(f'siam-pred -i {dd} -m {num_model} -o res')

#din = gdir(rd,['SEP_biopro_reslice|GLIOMRS|Lymphoma|Atlas)','vol'])
nnunet_siam_pred_job(din,model_num_list=[610])
fcsv, outdir_csv = gfile(din,'csv'), rd+'/csv_validat'
make_validation_csv(fcsv, outdir_csv, pred_regex='pred_DS610_nnAug3DSres',auto_DS='^DS712')
make_validation_csv(fcsv, outdir_csv, pred_regex='SuperSynt',auto_DS='^DSFree_lesion_remap')
make_validation_csv_HCP_repeate(fcsv[0],fcsv[1],outdir_csv, pred_regex='SuperSynt',auto_DS='^DSFree_lesion_remap')
make_validation_csv_HCP_repeate(fcsv[0],fcsv[0],outdir_csv, pred_regex='Fast',auto_DS='^DSFree_remap')
#et on enlève le short_xxx.csv car on veut la version realign
# pour T1 versus T2 ->  make_validation_csv_HCP_autoref(fcsvT1,fcsvT2 ...)
# pour T1 versus T1 realing
rd = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/HCP_test_retest_07mm_suj82/'
fcsvT1         = rd + 'vol_T1_07_sess1/HCP_test_retest_07mm_suj82_vol_T1_07_free_Ass_siam_nomask.csv'
fcsvT1_realing = rd + 'vol_T1_07_realign/Ses2_HCP_test_retest_07mm_suj82_vol_T1_07_free_Ass_siam_nomask.csv'
fcsvT1         = rd + 'vol_T1_07/HCP_test_retest_07mm_suj82_vol_T1_07_free_Ass_siam_nomask.csv' #remove FastSurfer
fcsvT1_realing = rd + 'vol_T2_07/HCP_test_retest_07mm_suj82_vol_T2_07_free_Ass_siam_nomask.csv'
for model_name in models:
    print(model_name)
    if model_name=='708':
        auto_DS = '^DS708'
    elif ('Synth' in model_name) or ('gouhfi' in model_name) or ('FastSur' in model_name):
        auto_DS='DSFree_lesion_remap'
    else:
        auto_DS = '^DS712'
    make_validation_csv_HCP_autoref(fcsvT1, fcsvT1_realing, outdir_csv, pred_regex=model_name,
                                        auto_DS=auto_DS, metrics_name=metrics)
    break

# manual change the remap, to get all tissus  DS712_remap_to_label_GTVas.csv
# pour Ass make_validation_csv_HCP_autoref(fcsvT1, fcsvT1_realing, outdir_csv, pred_regex='AssN',metrics_name=metrics,auto_DS='DSFree_lesion_remap')
#create repeate T1 on hcp arg not usefull allready in the funtion ...
for ff in fcsv:
    dn,fn = get_parent_path(ff)
    df = pd.read_csv(ff)
    df1, df2 = df[::2], df[1::2]
    do = gdir(dn,'cs_retest')[0]
    print(f'{do}/{fn}')
    df1.to_csv(f'{do}/Ses1_{fn}',index=False)
    df2.to_csv(f'{do}/Ses2_{fn}',index=False)
fcsvs = gfile(gdir(din,'cs_retest'),'csv',list_flaten = False)
for fcsv in fcsvs:
    make_validation_csv_HCP_repeate(fcsv[0],fcsv[1],outdir_csv, pred_regex='SuperSynt',auto_DS='^DSFree_lesion_remap')
    break

create_SynthSeg_job(din,model='super_invivo',jobdir=rd+'/job_pred/superS',prefix='SuperSynth')
create_AssN_job(fcsv,jobdir=rd+'/job_pred/AssN',)
#average nnunet models
fm = gfile(df,'final')
mw_list = [torch.load(mdic)['network_weights'] for mdic in fm[1:]]
mout = torch.load(fm[0])
mweight = mout['network_weights']
for kk in  mweight.keys() :
    print(kk)
    for mm in mw_list:
        mweight[kk] += mm[kk]
    mweight[kk] /= 5
mout['network_weights'] = mweight
torch.save(mout,'checkpoint_final.pth')

#data testing set
rd = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/'
job_rd = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/job_pred/'
dsep = gdir(rd+'SEP_biopro_reslice','.*'); dsep.pop(-2); dsep.pop(0)
dlym = gdir(rd+'Lymphoma','t1');
dgli = gdir(rd+'GLIOMRS','T[12]');
datl = gdir(rd+'Atlas','T1');

#remap syntseg  SEP 88*6 528  Lym 116  Gli 45*2=90 Atlas 655 -> total 1389
dsynth = gdir(dsep+dlym+dgli+datl,['Wmh'])
dsynth = gdir(din, ['SuperS','.*'])
ff = gfile(dsynth,'(^mni)|(^fake)|(^Synt)|(^inp)') ; #delete_file_list(ff)
fapar = gfile(dsynth,'^seg.*gz')
fref=[]
for ff in fapar:
    pp,name = get_parent_path(ff,2)
    fref += gfile(get_parent_path(pp)[0],name,opts={"items":1})

tmap = get_fastsurfer_remap(fapar[0],fcsv ='/network/iss/opendata/data/template/remap/free_remapV2.csv', index_col_remap=5) #Lession !
remap_filelist(fapar,tmap, fref=fref, prefix='remapSiam_',reslice_with_mrgrid=True)

dsynth = gdir(din, ['AssN','.*'])
fass = gfile(dsynth,'^native_structures.*nii.gz')
tmap = get_remapping('assn',tmap_index_col=[0,1])  #3 is 6 region but 5 is only 4
remap_filelist(fass,tmap, fref=None, prefix='remapSiam_')

make_validation_csv(fcsv, outdir_csv, pred_regex='AssN',auto_DS='^DS712')



#run some predictions
#SEP BIOPRO
jobdir=job_rd + 'SynthWHM'
create_SynthSeg_job(dsep,option=' ', lesion=True, prefix="SynthSegWmh",jobdir=jobdir)

#Lymphoma
jobdir=job_rd + 'SynthLym'
create_SynthSeg_job(dlym,option=' ', lesion=True, prefix="SynthSegWmh",jobdir=jobdir)

#GlioMRS
jobdir=job_rd + 'SynthGLIO'
create_SynthSeg_job(dgli,option=' ', lesion=True, prefix="SynthSegWmh",jobdir=jobdir)

#Atlas
jobdir=job_rd + 'SynthAtl'
create_SynthSeg_job(datl,option=' ', lesion=True, prefix="SynthSegWmh",jobdir=jobdir)

