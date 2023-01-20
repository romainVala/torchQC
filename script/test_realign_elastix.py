import SimpleITK as sitk
import torchio as tio
import numpy as np
np.set_printoptions(2)

from util_affine import *



#tes single affine
sdata = tio.datasets.Colin27()
sdata.pop('head'); sdata.pop('brain')


#sdata.t1.save('orig.nii')
P = np.zeros(6)
for k in range(0,6):
    #P = np.zeros(6)
    P[k] = (k-2)*10 + 5
    Euler_angle, Translation = P[3:] , P[:3] ;

    taff = tio.Affine([1,1,1],Euler_angle, Translation, center='image');  img_center_ras = np.array(sdata.t1.get_center())
    taff = tio.Affine([1,1,1],Euler_angle, Translation, center='origin');img_center_ras=np.array([0. ,0., 0.])
    img_center_lps = ras_to_lps_vector(img_center_ras)

    affitk = itk_euler_to_affine(Euler_angle, Translation, img_center_ras, make_RAS=False, set_ZYX=False)
    #affitk = itk_euler_to_affine(Euler_angle, Translation, img_center_lps, make_RAS=True)  #identical !
    #aff = spm_matrix(np.hstack([P, [1, 1, 1, 0, 0, 0]]), order=1, set_ZXY=True, rotation_center= img_center_ras)
    aff = get_matrix_from_euler_and_trans(P, rot_order = 'yxz', rotation_center=img_center_ras)
    if 1==1:
        srot = taff(sdata)
        sname = f'tio_A_ind{k}.nii'
        srot.t1.save(sname)

        aff_elastix, elastix_trf = ElastixRegister(sdata.t1, srot.t1)
        #aff_elastix, elastix_trf = ElastixRegister( srot.t1, sdata.t1); aff_elastix = np.linalg.inv(aff_elastix)

        aff_elastix[abs(aff_elastix)<0.0001]=0
        if np.allclose(affitk, aff_elastix, atol=1e-1):
            print(f'k:{k} almost equal ')
        else:
            #print(f'k:{k} transform tioAff (then elastix) \n {affitk} \n Elastix {aff_elastix}  ')
            print(f'k:{k} tioAff - elastix) \n {affitk - aff_elastix}  ')

        #write coreg image
        aff_elastix, elastix_trf = ElastixRegister( srot.t1, sdata.t1)

        datacoreg = elastix_trf.GetResultImage()
        np_array = sitk.GetArrayFromImage(datacoreg)
        np_array = np_array.transpose()  # ITK to NumPy
        srot.t1.data[0]  = torch.tensor(np_array);  srot.t1.save('move_back.nii')

    if 1==1: #compare euler to affine itk and spm  -> this works in all conditions
        if np.allclose(affitk, aff, atol=1e-6):
            print(f'k:{k} spm==tioAff  ')
        else:
            print(f'k:{k} transform tioAff (then spm) \n {affitk} \n {aff}  ')
            print(f'k:{k} spm - tioAff ) \n {aff - affitk}  ')

    P_reverse_ideal = get_euler_and_trans_from_matrix(affitk, rotation_center=img_center_ras)
    P_reverse_elastix = get_euler_and_trans_from_matrix(aff_elastix, rotation_center=img_center_ras)

    print(f'input param in {P} \n revers trans  {P_reverse_ideal} \n revers elast {P_reverse_elastix}')


import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
from transforms3d.euler import euler2mat, mat2euler

img_center_ras = [0,0,0]
P = np.zeros(6)
for k in range(3,6):
    #P = np.zeros(6)
    P[k] = (k-2)*10
    Euler_angle, Translation = P[3:] , P[:3] ;
    aff = spm_matrix(np.hstack([P, [1, 1, 1, 0, 0, 0]]), order=1, set_ZXY=False)
    if 0==1:
        aff = itk_euler_to_affine(Euler_angle, Translation, img_center_ras, make_RAS=False, set_ZYX=True)
        aff_ex = pyrot.active_matrix_from_extrinsic_euler_xyz(np.deg2rad(P[3:6]))
        aff_ex = rrr_active_matrix_from_extrinsic_euler_xyz(np.deg2rad(P[3:6]))
        aff_ex = euler2mat(np.deg2rad(P[3]), np.deg2rad(P[4]), np.deg2rad(P[5]), axes='sxyz')
    else:
        aff = itk_euler_to_affine(Euler_angle, Translation, img_center_ras, make_RAS=False, set_ZYX=False)
        aff_ex = euler2mat(np.deg2rad(P[4]), np.deg2rad(P[3]), np.deg2rad(P[5]), axes='syxz')
        #euler2mat(np.deg2rad(P[3]), np.deg2rad(P[4]), np.deg2rad(P[5]), axes='szyx')

    aff_ex = pytr.transform_from(aff_ex, P[:3])

    if np.allclose(aff, aff_ex, atol=1e-5):
        print(f'{k} == ext')
    else:
        print(f'{k} differ')
        print(f'{aff}\n{aff_ex}')


#Notes

# aff_ex = pyrot.matrix_from_euler_xyz(np.deg2rad(P[3:6]))
# ==
# aff = spm_matrix(np.hstack([P, [1, 1, 1, 0, 0, 0]]), order=1, set_ZXY=False)



# ITK convention looks like active_matrix_from_angle

theta_x = np.deg2rad(20)
theta_y = np.deg2rad(30)
theta_z = np.deg2rad(40)

sitk_R = sitk.Euler3DTransform()
sitk_R.SetRotation(theta_x, theta_y, theta_z)

Rx = np.array([[1, 0, 0],
               [0, np.cos(theta_x), -np.sin(theta_x)],
               [0, np.sin(theta_x), np.cos(theta_x)]])

Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
               [0, 1, 0],
               [-np.sin(theta_y), 0, np.cos(theta_y)]])

Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
               [np.sin(theta_z), np.cos(theta_z), 0],
               [0, 0, 1]])

print('ZXY:')
print(Rz.dot(Rx).dot(Ry))
print(sitk_R.GetMatrix())

print('ZYX:')
sitk_R.SetComputeZYX(True)
print(Rz.dot(Ry).dot(Rx))
print(sitk_R.GetMatrix())

# aff_ex = pyrot.active_matrix_from_extrinsic_euler_yxz(np.deg2rad(P[3:6]))
# aff_ex = pyrot.active_matrix_from_extrinsic_euler_zxy(np.deg2rad(P[3:6]))

# aff_ex = pyrot.active_matrix_from_intrinsic_euler_zxy(np.deg2rad(P[3:6]))
# aff_ex = pyrot.active_matrix_from_intrinsic_euler_yxz(np.deg2rad(P[3:6]))
# aff_ex = pyrot.active_matrix_from_intrinsic_euler_xyz(np.deg2rad(P[3:6]))

# aff_ex = pyrot.matrix_from_euler_zyx(np.deg2rad(P[3:6]))
# aff_ex = pyrot.matrix_from_euler_xyz(np.deg2rad(P[3:6]))

def active_matrix_from_angle(basis, angle):
    """Compute active rotation matrix from rotation about basis vector.

    Parameters
    ----------
    basis : int from [0, 1, 2]
        The rotation axis (0: x, 1: y, 2: z)

    angle : float
        Rotation angle

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix

    Raises
    ------
    ValueError
        If basis is invalid
    """
    c = np.cos(angle)
    s = np.sin(angle)

    if basis == 0:
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, c, -s],
                      [0.0, s, c]])
    elif basis == 1:
        R = np.array([[c, 0.0, s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0, c]])
    elif basis == 2:
        R = np.array([[c, -s, 0.0],
                      [s, c, 0.0],
                      [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Basis must be in [0, 1, 2]")

    return R

def rrr_active_matrix_from_extrinsic_euler_yxz(e):

    alpha, beta, gamma = e
    R = active_matrix_from_angle(2, gamma).dot(
        active_matrix_from_angle(0, alpha)).dot(
        active_matrix_from_angle(1, beta)
        )
    return R

def rrr_active_matrix_from_extrinsic_euler_xyz(e):
    #no change from original
    alpha, beta, gamma = e
    R = active_matrix_from_angle(2, gamma).dot(
        active_matrix_from_angle(1, beta)).dot(
        active_matrix_from_angle(0, alpha))
    return R

def get_matrix_from_euler_and_trans_2(P, rot_order = 'xyz', rotation_center=None):
    rot = np.deg2rad(P[3:6])
    trans = P[:3]
    if rot_order=='xyz':
        aff = rrr_active_matrix_from_extrinsic_euler_xyz(rot)
    elif rot_order=='yxz':
        aff = rrr_active_matrix_from_extrinsic_euler_yxz(rot)
    else:
        raise(f'rotation order {rot_order} not implemented')

    aff = pytr.transform_from(aff, trans)

    if rotation_center is not None:
        aff = change_affine_rotation_center(aff, rotation_center)
    return aff
