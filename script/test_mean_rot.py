import numpy as np
import quaternion as nq
from dual_quaternions import DualQuaternion
from util_affine import  * # spm_matrix, spm_imatrix


np.random.seed(4)
nb_mean=500
euler_mean, euler_choral, euler_exp, euler_pol, euler_slerp, euler_qr_slerp = np.zeros((nb_mean,3)), np.zeros((nb_mean,3)), np.zeros((nb_mean,3)), np.zeros((nb_mean,3)), np.zeros((nb_mean,3)), np.zeros((nb_mean,3))
for i in range(nb_mean):
    #rot_euler = np.random.normal(size=(10, 3),loc=20,scale=5) #in degree
    rot_euler = np.random.uniform(-20,20,size=(10, 3)) #in degree
    #print(f'min {np.min(rot_euler)} max {np.max(rot_euler)}')
    aff_list = get_affine_rot_from_euler(rot_euler) #4*4 affine matrix
    qr_list = [ nq.from_rotation_matrix(aff) for aff in aff_list]  #unit quaternion

    euler_mean[i,:] = np.mean(rot_euler,axis=0)

    qr_cm =  nq.mean_rotor_in_chordal_metric(qr_list); #print(get_info_from_quat(qr_cm)); print(get_euler_for_qr(qr_cm))
    euler_choral[i,:] = get_euler_from_qr(qr_cm)

    aff_exp_mean = exp_mean_affine(aff_list);     #print(spm_imatrix(aff_exp_mean)[3:6])
    euler_exp[i,:]  = get_euler_from_affine(aff_exp_mean)

    aff_polar_mean = polar_mean_affin(aff_list); #print(spm_imatrix(aff_exp_mean)[3:6])
    euler_pol[i,:] = get_euler_from_affine(aff_polar_mean)

    qr_mean = qr_slerp_mean(qr_list)
    euler_qr_slerp[i,:] = get_euler_from_qr(qr_mean)

print(f'max diff euler between exp matrix and euler mean {np.max(np.abs(euler_exp - euler_mean))}')
print(f'max diff euler between polar matrix and euler mean {np.max(np.abs(euler_pol - euler_mean))}')
print(f'max diff euler between exp matrix and polar {np.max(np.abs(euler_exp - euler_pol))}')
print('####')
print(f'max diff euler between slerp and euler mean {np.max(np.abs(euler_qr_slerp - euler_mean))}')
print(f'max diff euler between slerp and exp  {np.max(np.abs(euler_qr_slerp - euler_exp))}')
print(f'max diff euler between slerp and polar  {np.max(np.abs(euler_qr_slerp - euler_pol))}')
print('####')
print(f'max diff euler between choral and euler mean {np.max(np.abs(euler_choral - euler_mean))}')
print(f'max diff euler between slerp and choral  {np.max(np.abs(euler_qr_slerp - euler_choral))}')
print(f'max diff euler between exp matrix and choral {np.max(np.abs(euler_exp - euler_choral))}')
print(f'max diff euler between choral and polar {np.max(np.abs(euler_choral - euler_pol))}')
#
# #if euler [-5 5]
# max diff euler between exp matrix and euler mean 0.06626835745519921
# max diff euler between polar matrix and euler mean 0.06595075004752815
# max diff euler between exp matrix and polar 0.0036721538198414283
# ####
# max diff euler between slerp and euler mean 0.06589127323023924
# max diff euler between slerp and exp  0.0016206341305240457
# max diff euler between slerp and polar  0.0036587992686001325
# ####
# #if euler [-5 5]
# max diff euler between exp matrix and euler mean 1.5662635805067207
# max diff euler between polar matrix and euler mean 1.7107179786318594
# max diff euler between exp matrix and polar 0.2453864630127489
# ####
# max diff euler between slerp and euler mean 1.5392422583833583
# max diff euler between slerp and exp  0.11185358484109642
# max diff euler between slerp and polar  0.29266563097344833
# ####


#difference / mean
#from test_mean_rot import get_euler_from_dq, get_euler_from_qr, get_info_from_dq,  get_affine_rot_from_euler,paw_quaternion

rot_euler = np.random.uniform(-20,20,size=(2, 3)) #in degree
aff1, aff2 = get_affine_rot_from_euler(rot_euler)
aff1=spm_matrix([0,0,0,0,10,0,1,1,1,0,0,0],order=1)
aff2=spm_matrix([0,0,0,0,20,0,1,1,1,0,0,0],order=1)

dq1=DualQuaternion.from_homogeneous_matrix(aff1);dq2=DualQuaternion.from_homogeneous_matrix(aff2)
dqmean = DualQuaternion.sclerp(dq1,dq2,0.5); print(get_info_from_dq(dqmean)); print(get_euler_from_dq(dqmean))
np.mean(rot_euler,axis=0)
#domage, ca marche pas !
dqmean2 = dq1.pow(0.5)*dq2.pow(0.5);  print(get_info_from_dq(dqmean2)); print(get_euler_from_dq(dqmean2))

dqdiff = dq1.inverse()*dq2 #dq1*dq2.quaternion_conjugate()
[print(f'{k}: {val}') for k,val in get_info_from_dq(dqdiff).items()]; print(get_euler_from_dq(dqdiff))
#dqmean3 = dqdiff.pow(0.5)*dq1.quaternion_conjugate(); get_info_from_dq(dqmean3); print(get_euler_from_dq(dqmean3))
dqmean3 = dq1 * dqdiff.pow(0.5); print(get_info_from_dq(dqmean3)); print(get_euler_from_dq(dqmean3))

qr1 = nq.from_rotation_matrix(aff1); qr2 = nq.from_rotation_matrix(aff2); get_euler_from_qr(qr1)
qrmean = paw_quaternion(qr1, 0.5) * paw_quaternion(qr2, 0.5) ; print(get_info_from_quat(qrmean))
qrmc =  nq.mean_rotor_in_chordal_metric([qr1, qr2]); print(get_info_from_quat(qrmc)); get_euler_from_qr(qrmc)



dq1 = DualQuaternion.from_translation_vector([0,0,1])
dq2 = DualQuaternion.from_translation_vector([0,1,0])

l1=[0,0,1]; o1 = [10,10,0]; m1 = np.cross(l1,o1); theta1 = np.deg2rad(0); disp1=5;
l2=[0,1,0]; o2 = [10,10,0]; m2 = np.cross(l2,o2); theta2 = np.deg2rad(0); disp2=5;
dq1 = DualQuaternion.from_screw(l1, m1, theta1, disp1) #resultant aff trans is displacement of the origine from screw axis rot
dq2 = DualQuaternion.from_screw(l2, m2, theta2, disp2) #resultant aff trans is displacement of the origine from screw axis rot

dqmean2 = dq1.pow(0.5)*dq2.pow(0.5);  get_info_from_dq(dqmean2)

(np.array(l1)*disp1 + np.array(l2)*disp2)/2


#test random rotation matrix
Nmat=1500
rot_euler = np.random.uniform(-20,20,size=(Nmat, 3)) #in degree
aff_eul = [spm_matrix( np.hstack([[0,0,0], r, [1,1,1,0,0,0]]), order=1) for r in rot_euler]
#does not work   ...   aff_eul = get_affine_rot_from_euler(rot_euler)
#aff_lis = [random_rotation_matrix(20/360) for i in range(Nmat)]
aff_lis = [random_rotation(20) for i in range(Nmat)]
#aff_lis = [random_rotation_matrix() for i in range(1000)]

theta = np.zeros(len(aff_lis)); d = np.zeros(len(aff_lis))
s_ax_dir = np.zeros((3,len(aff_lis))); m = np.zeros((3,len(aff_lis)))
for ii,aff in enumerate(aff_eul):
    s_ax_dir[:,ii], m[:,ii], theta[ii], d[ii] = get_screw_from_affine(aff)

for ii,aff in enumerate(aff_lis):
    s_ax_dir[:,ii], m[:,ii], theta[ii], d[ii] = get_screw_from_affine(aff)

plt.figure(); plt.hist(theta, bins=100)
fig = plt.figure();ax = plt.axes(projection ='3d');plt.xlabel('x') ;plt.ylabel('y')
ax.scatter(s_ax_dir[0,:], s_ax_dir[1,:], s_ax_dir[2,:])
#X, Y, Z = zip(*origin_pts.T); U,V,W = zip(*axiscrew.T*5); ax.quiver(X, Y, Z, U, V, W)
