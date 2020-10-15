import commentjson as json

file = '/home/romain.valabregue/datal/PVsynth/validation_set/RES_1mm/config_all.json'
file = 'config_all.json'
with open(file) as f:
    conf_all =  json.load(f)

label_transfo = conf_all['transform']['val_transforms'][0]
noise_transfo = conf_all['transform']['val_transforms'][1]
label_list = conf_all['data']['labels']

gaussian_mean = label_transfo['attributes']['mean'] #label_transfo['attributes']['gaussian_parameters']

wm_label = ['BrStem', 'WM',  'cereb_WM' ]
gm_label = ['GM', 'cereb_GM', 'L_Accu', 'L_Amyg', 'L_Caud', 'L_Hipp', 'L_Pall', 'L_Puta', 'L_Thal', 'R_Accu', 'R_Amyg', 'R_Caud', 'R_Hipp', 'R_Pall', 'R_Puta', 'R_Thal', ]
csf_label = ['CSF',  ]
bg_label = ['skin', 'skull', 'background']

wm_label_index = [label_list.index(label) for label in wm_label]
gm_label_index = [label_list.index(label) for label in gm_label]
csf_label_index = [label_list.index(label) for label in csf_label]
bg_label_index = [label_list.index(label) for label in bg_label]

gm_values = [0.3, 0.6, 0.9]
csf_values = [0.2, 1]
wm_values = [1, 0.2]
snr_level = [0.01, 0.05, 0.1]

gs = dict()
gs['transform.val_transforms.0.attributes.mean'] =  dict(prefix='dataS', values=[], names=[])
gs['transform.val_transforms.1.attributes.std'] = dict(prefix='SNR', values=[], names=[])

for snr_value in snr_level:
    gs['transform.val_transforms.1.attributes.std']['values'].append([snr_value, snr_value])
    gs['transform.val_transforms.1.attributes.std']['names'].append(f'{int(1/snr_value)}')

for csf_mean, wm_mean in zip(csf_values, wm_values):
    for gm_mean in gm_values:
        gaussian_mean = label_transfo['attributes']['mean'].copy()
        for ll in wm_label_index:
            gaussian_mean[ll] = wm_mean
        for ll in gm_label_index:
            gaussian_mean[ll] = gm_mean
        for ll in csf_label_index:
            gaussian_mean[ll] = csf_mean
        print(gaussian_mean)
        gs['transform.val_transforms.0.attributes.mean']['values'].append(gaussian_mean)
        gs['transform.val_transforms.0.attributes.mean']['names'].append(f'GM{int(gm_mean*10)}_WM{int(wm_mean*10)}_C{int(csf_mean*10)}')


#write separate json file
res =  '/home/romain.valabregue/datal/PVsynth/validation_set/RES_1mm/'
res=''
for k in conf_all.keys():
    filename = res+k+'.json'
    with open(filename, 'w') as file:
        json.dump(conf_all[k], file, indent=4, sort_keys=True)

filename = res+'grid_search.json'
with open(filename, 'w') as file:
    json.dump(gs, file, indent=4, sort_keys=True)


# generting jobs for validation
from utils_file import gdir, gfile, get_parent_path
f = gfile('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/training/RES_1mm_tissue/pve_synth_data_92_common_noise_no_gamma/results_cluster',
          'model.*tar')
d='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/training/RES_1mm_tissue/'
dres = gdir(d,['.*','result'])
dresname = get_parent_path(dres,level=2)[1]
dresname = [dd.split('_')[0] + '_' + dd.split('_')[1] for dd in dresname]

for one_res, resn in zip(dres, dresname):
    f = gfile(one_res,'model.*tar')
    fname = get_parent_path(f)[1]
    for ff in f:
        print('\"{}\",'.format(ff))

for one_res, resn in zip(dres, dresname):
    f = gfile(one_res,'model.*tar')
    fname = get_parent_path(f)[1]

    for ff in fname:
        fname_ep = ff.split('_')[1]

        print('\"{}_{}\",'.format(resn,fname_ep))
