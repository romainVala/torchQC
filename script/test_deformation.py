#from fernando https://gist.github.com/fepegar/b723d15de620cd2a3a4dbd71e491b59d
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torchio as tio
from torchio.data.io import nib_to_sitk
from skimage import measure
i = tio.LabelMap('/data/romain/template/MIDA_v1.0/MIDA_v1_voxels/mida_merge_v2_target.nii')
suj = tio.Subject(lab=i)
mean_val = [[0,0],[0.5,0.5],[0.9,0.9],[0.1,0.1],[0.1,0.1],[0.45,0.45],[0.7,0.7],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
tlab = tio.RandomLabelsToImage(image_key='t1',mean= mean_val,default_std = [0.01, 0.1],label_key='lab',
                               ignore_background=True)
#sujt = tlab(suj)
#sujt.t1.save('/data/romain/template/MIDA_v1.0/MIDA_v1_voxels/mida_brain_T1.nii')
ta = tio.Compose([tlab, tio.ToCanonical(), tio.Crop((11,11,40,40,151,30)), tio.Resample(target=1)])
sujt = ta(suj)
sujt.t1.save('/data/romain/template/MIDA_v1.0/MIDA_v1_voxels/brain_T1mm_mida.nii')


#from fernando elastic gist
N = 256
grid_spacing = 64

grid = sitk.GridSource(
    outputPixelType=sitk.sitkFloat32,
    size=(N, N),
    sigma=(0.5, 0.5),
    gridSpacing=(grid_spacing, grid_spacing),
    gridOffset=(0, 0),
    spacing=(1, 1),
)

array = sitk.GetArrayViewFromImage(grid)
fig, ax = plt.subplots(dpi=150)
ax.imshow(
    array,
    interpolation='hamming',
)

ctrl_pts = 7, 7
fix_edges = 2

ctrl_pts = np.array(ctrl_pts, np.uint32)
SPLINE_ORDER = 3
mesh_size = ctrl_pts - SPLINE_ORDER
transform = sitk.BSplineTransformInitializer(grid, mesh_size.tolist())
params = transform.GetParameters()

grid_shape = *ctrl_pts, 2

max_displacement = 50
uv = np.random.rand(*grid_shape) - 0.5  # [-0.5, 0.5)
uv *= 2  # [-1, 1)
uv *= max_displacement

#uv *= 0
#uv[3, 2] = -50, 0  # indices are x, y

#std = 20
#uv = np.random.randn(*grid_shape)
#uv *= std


for i in range(fix_edges):
    uv[i, :] = 0
    uv[-1 - i, :] = 0
    uv[:, i] = 0
    uv[:, -1 - i] = 0

transform.SetParameters(uv.flatten(order='F').tolist())
x_coeff, y_coeff = transform.GetCoefficientImages()
grid_origin = x_coeff.GetOrigin()
grid_spacing = x_coeff.GetSpacing()

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(grid)
resampler.SetTransform(transform)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0.5)
resampler.SetOutputPixelType(sitk.sitkFloat32)
resampled = resampler.Execute(grid)

array = sitk.GetArrayViewFromImage(resampled)

fig, ax = plt.subplots(dpi=150)
ax.grid()
ax.imshow(
    array,
    #extent=[
    #    0, resampled.GetSize()[1] * resampled.GetSpacing()[1],
    #    0, resampled.GetSize()[0] * resampled.GetSpacing()[0],
    #],
    interpolation='hamming',
)

x = np.linspace(grid_origin[0], grid_origin[0] + (ctrl_pts[0] - 1) * grid_spacing[0], ctrl_pts[0])
y = np.linspace(grid_origin[1], grid_origin[1] + (ctrl_pts[1] - 1) * grid_spacing[1], ctrl_pts[1])
xx, yy = np.meshgrid(x, y)
u, v = uv[..., 0].T, uv[..., 1].T
ax.quiver(xx, yy, -u, -v, color='red',
          #width=0.0075,
          units='xy', angles='xy', scale_units='xy', scale=1)
ax.scatter(xx, yy, s=1);


toDisplacementFilter = sitk.TransformToDisplacementFieldFilter()
toDisplacementFilter.SetReferenceImage(grid)

displacementField = toDisplacementFilter.Execute(transform)

i = tio.LabelMap('/data/romain/template/MIDA_v1.0/MIDA_v1_voxels/mida_merge_v2_target.nii')
csfmask = i.data.numpy().astype(np.float64)
csf = np.ones_like(csfmask); csf[csfmask==4] = 2
csf_orig = np.zeros_like(csfmask); csf_orig[csfmask==4] = 1
csf_orig_im =  nib_to_sitk(csf_orig, i.affine)
csf_im = nib_to_sitk(csf, i.affine)
label_im = nib_to_sitk(i.data, i.affine )

import disptools.displacements as dsp
displacement = dsp.displacement(csf_im, epsilon=0.01, it_max=200)

tf = sitk.DisplacementFieldTransform(3)
param = list(displacement.GetSize()) + list(displacement.GetOrigin()) +list(displacement.GetSpacing())+ list( displacement.GetDirection())
tf.SetFixedParameters(param)
tf.SetInterpolator(sitk.sitkNearestNeighbor)
dd = sitk.GetArrayFromImage(displacement)
tf.SetParameters(dd.flatten().astype(np.float64))

reference=csf_im
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(reference)
resampler.SetTransform(tf)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetDefaultPixelValue(0)
resampler.SetOutputPixelType(sitk.sitkFloat32)
resampled = resampler.Execute(csf_orig_im)
sitk.WriteImage(resampled,'csf_out.nii')
resampled = resampler.Execute(label_im)
for i in range(5):
    resampled = resampler.Execute(resampled)

sitk.WriteImage(resampled,'label_out55.nii')

#result, _ = self.sitk_to_nib(resampled)
sitk.WriteImage(csf_im,'csf_ini.nii')



i = tio.LabelMap('/data/romain/template/MIDA_v1.0/MIDA_v1_voxels/mida_merge_v2_target.nii')
csfmask = i.data.numpy().astype(np.int8)[0]
csf_orig = np.zeros_like(csfmask); csf_orig[csfmask==4] = 1
csf_orig_im =  nib_to_sitk(i.data, i.affine)

# Use marching cubes to obtain the surface mesh of these ellipsoids
#verts, faces, normals, values = measure.marching_cubes_lewiner(csf_orig)
verts, faces, normals, values = measure.marching_cubes(csf_orig)
disp_3D = np.zeros(list(csf_orig.shape)+ [3])
nb_3D = np.zeros(list(csf_orig.shape))

for point, normal  in zip(verts, normals):
    #print(point)
    disp_3D[int(point[0]), int(point[1]), int(point[2]), 0 ] += normal[2]
    disp_3D[int(point[0]), int(point[1]), int(point[2]), 1 ] += normal[0]
    disp_3D[int(point[0]), int(point[1]), int(point[2]), 2 ] += normal[1]
    nb_3D[int(point[0]), int(point[1]), int(point[2]) ] += 1
nb_3D[nb_3D==0] = 1
disp_3D[...,0] /= nb_3D;disp_3D[...,1] /= nb_3D;disp_3D[...,2] /= nb_3D
disp_3D*=20

dd = np.transpose(disp_3D, (3,0,1,2))
img_disp = tio.ScalarImage(tensor=torch.tensor(dd), affine=i.affine)
tior = tio.Resample(target=1)
timg = tior(img_disp)
timg.save('rdisp4.nii')
sitk_img =  nib_to_sitk(timg.data, timg.affine)
sitk_img =  nib_to_sitk(img_disp.data, img_disp.affine)
sitk.WriteImage(sitk_img,'ssfull.nii')
disp_array = sitk.GetArrayFromImage(sitk_img).transpose()
#disp_array = np.transpose(disp_array,(1,2,3,0))

disp_shape = [ii-3 for ii in timg.data[0].shape]

transform = sitk.BSplineTransformInitializer(csf_orig_im, disp_shape)
#transform.SetParameters(disp_field.flatten(order='F').tolist())
transform.SetParameters(disp_array.flatten(order='F').tolist())

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(csf_orig_im)
resampler.SetTransform(transform)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetDefaultPixelValue(0)
resampler.SetOutputPixelType(sitk.sitkInt8)
resampled = resampler.Execute(csf_orig_im)

sitk.WriteImage(resampled,'deform_label.nii.gz')
