import scipy.linalg as scl
import numpy as np
import quaternion as nq
from dual_quaternions import DualQuaternion
from util_affine import  spm_matrix, spm_imatrix
import numpy.linalg as npl

def get_screw_from_affine(affine):
    dq = DualQuaternion.from_homogeneous_matrix(affine)
    s_ax_dir, m, theta, d = dq.screw()
    theta = np.rad2deg(theta)
    return s_ax_dir, m, theta, d
def get_info_from_quat(q):
    angle = get_rotation_angle(q)
    ax = get_rotation_axis(q)
    return dict(angle=angle, ax=ax)
def get_info_from_dq(dq, verbose=False):
    l, m, tt, dd = dq.screw()#(rtol=1e-5)
    theta = np.rad2deg(tt); disp = dd
    if npl.norm(l)<1e-10:
        origin_pts = [0, 0, 0]
    else:
        #origin_pts = np.cross(m,l) # because norm of l is 1
        origin_pts = np.cross(l,m) # change to match set get and have the origine where it should
        #so with this the set is done from orig_pos with  m = np.cross(orig_pos,l);
        #from spatialmath         return np.cross(self.v, self.w) / np.dot(self.w, self.w)    # where V (3-vector) is the moment and W (3-vector) is the line direction.
        #origin_pts = np.cross(m,l)/np.dot(l,l)
    line_distance = npl.norm(m)
    # from spatialmath        return math.sqrt(np.dot(self.v, self.v) / np.dot(self.w, self.w) )
    #line_dist2 = np.sqrt(np.dot(m,m)/np.dot(l,l))
    trans = dq.translation()
    res = dict(l=l, m=m, origin_pts=origin_pts, line_dist=line_distance, disp=disp, theta=theta, trans=trans)
    if verbose:
        for k in res:
            print(f'{k}: {res[k]}')
    return res

#let's explore quaternion
def get_rotation_angle2(q):
    angle = np.linalg.norm(nq.as_rotation_vector(q))
    if angle>np.pi:
        angle = angle - 2*np.pi
    return np.rad2deg(angle)
def get_rotation_angle(q):
    qa=nq.as_float_array(q)
    angle = 2*np.arctan2(np.sqrt(qa[1]**2+qa[2]**2+qa[3]**2),qa[0])
    if angle>np.pi:
        angle = angle - 2*np.pi
    return np.rad2deg(angle)
def get_rotation_axis(q):
    qa = nq.as_float_array(q)
    return qa[1:] / np.sqrt(qa[1]**2+qa[2]**2+qa[3]**2)
def paw_quaternion(qr, exponent):
    rot_vector = nq.as_rotation_vector(qr)
    #theta = np.linalg.norm(rot_vector)
    theta = 2 * np.arccos(nq.as_float_array(qr)[0] )  #equivalent
    s0 = rot_vector / np.sin(theta / 2)

    quaternion_scalar = np.cos(exponent*theta/2)
    quaternion_vector = s0 * np.sin(exponent*theta/2)

    return nq.as_quat_array( np.hstack([quaternion_scalar, quaternion_vector]) )
def exp_mean_affine(aff_list, weights=None):
    if weights is None:
        weights = np.ones(len(aff_list))
    #normalize weigths
    weights = weights / np.sum(weights)

    Aff_mean = np.zeros((4, 4))
    for aff, w in zip(aff_list, weights):
        Aff_mean = Aff_mean + w*scl.logm(aff)
    Aff_mean = scl.expm(Aff_mean)
    return Aff_mean
def polar_mean_affin(aff_lis, weights=None):
    if weights is None:
        weights = np.ones(len(aff_list))
    #normalize weigths
    weights = weights / np.sum(weights)
    Aff_Euc_mean = np.zeros((3, 3))
    for aff, w in zip(aff_list, weights):
        Aff_Euc_mean = Aff_Euc_mean + w * aff[:3, :3]

    Aff_mean = np.eye(4)
    Aff_mean[:3,:3] = scl.polar(Aff_Euc_mean)[0]
    return Aff_mean
def dq_slerp_mean(dq_list):
    c_num, c_deno = 1, 2
    for ii, dq in enumerate(dq_list):
        if ii==0:
            res_mean = dq
        else:
            t = 1 - c_num/c_deno
            res_mean = DualQuaternion.sclerp(res_mean, dq, t) #res_mean * c_num + dq) / c_deno
            c_num+=1; c_deno+=1
    return res_mean
def qr_slerp_mean(qr_list):
    c_num, c_deno = 1, 2
    for ii, qr in enumerate(qr_list):
        if ii==0:
            res_mean = qr
        else:
            # qr_mult = res_mean*qr
            # if nq.as_float_array(qr_mult)[0] < 0:
            #     #print("changing sign !!")
            #     res_mean = res_mean

            t = 1 - c_num/c_deno
            res_mean = nq.slerp(res_mean, qr, 0, 1, t) #res_mean * c_num + dq) / c_deno
            c_num+=1; c_deno+=1
    return res_mean

def my_mean(x_list):
    #just decompose the mean as a recursiv interpolation between 2 number, (to be extend to slerp interpolation)
    c_num, c_deno = 1, 2
    for ii, x in enumerate(x_list):
        if ii==0:
            res_mean = x
        else:
            t = 1 - c_num/c_deno
            res_mean = np.interp(t, [0,1], [res_mean, x])
            #res_mean = (res_mean * c_num + x) / c_deno
            c_num+=1; c_deno+=1
    return res_mean
x = np.random.random(100)
np.mean(x)-my_mean(x)

#to and from euler with different representation using spm_matrix euler conversion
def get_affine_rot_from_euler(e_array):
    aff_list=[]
    for r in e_array:
        aff_list.append(spm_matrix([0, 0, 0, r[0],r[1],r[2], 1, 1, 1, 0, 0, 0], order=1) )
        #aff_list.append(spm_matrix([0, 0, 0, r[0], 0,0, 1, 1, 1, 0, 0, 0], order=1))
    return aff_list
def get_euler_from_qr(qr):
    qraff = np.eye(4, 4);
    qraff[:3, :3] = nq.as_rotation_matrix(qr);
    return spm_imatrix(qraff)[3:6]
def get_euler_from_dq(dq):
    return spm_imatrix(dq.homogeneous_matrix())[3:6]
def get_euler_from_affine(aff):
    return spm_imatrix(aff)[3:6]

# same here : to and from euler with different representation but using quaternion euler convention

def get_affine_rot_from_euler(e_array):
    print('waring this does not work properly, with negativ euler angles ... ')
    aff_list=[]
    for r in e_array:
        r = np.deg2rad(r)
        qr = nq.from_euler_angles(r)
        aff = np.eye(4, 4);
        aff[:3,:3] = nq.as_rotation_matrix(qr)
        aff_list.append(aff)
    return aff_list
def get_modulus_euler_in_degree(euler):
    euler = np.rad2deg(euler)
    for index, e in enumerate(euler):
        if e < -180:
            euler[index] = e + 360
        if e > 180:
            euler[index] = e -360
    return euler
def get_euler_from_qr(qr,  mode='spm'):
    if mode=='spm':
        aff = np.eye(4,4)
        aff[:3,:3] = nq.as_rotation_matrix(qr)
        P = spm_imatrix(aff)
        return P[3:6]
    else:
        return get_modulus_euler_in_degree(nq.as_euler_angles(qr))
def get_euler_from_dq(dq, mode='spm'):
    if mode=='spm':
        P = spm_imatrix(dq.homogeneous_matrix())
        return P[3:6]
    else:
        qr = nq.from_rotation_matrix(dq.homogeneous_matrix())
        return get_modulus_euler_in_degree(nq.as_euler_angles(qr))
def get_euler_from_affine(aff, mode='spm'):
    if mode=='spm':
        P = spm_imatrix(aff)
        return P[3:6]
    else:
        qr = nq.from_rotation_matrix(aff)
        return get_modulus_euler_in_degree(nq.as_euler_angles(qr))

def random_rotation(amplitude = 360):
    """
    from_rotation_vector(rot):
    rot : (Nx3) float array
        Each vector represents the axis of the rotation, with norm
        proportional to the angle of the rotation in radians.
    """
    #get random orientation
    #V = np.random.uniform(-1,1,3) #np.random.rand(3)
    V = np.random.normal(size=3) #to get uniform orientation ...
    V = V / npl.norm(V) #warning if small norm ... ?
    theta = np.random.rand(1) * amplitude / 360 * np.pi * 2
    V = V * theta
    quat = nq.from_rotation_vector(V)
    aff = np.eye(4, 4);
    aff[:3,:3] = nq.as_rotation_matrix(quat)
    return aff

def random_rotation_matrix(amplitude=1, randgen=None):
    """
    Bof bof, I do not understand the resulting theta we get if we restric amplitude (if not seems ok)
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    *  R A N D _ R O T A T I O N      Author: Jim Arvo, 1991                  *
    *                                                                         *
    *  This routine maps three values (x[0], x[1], x[2]) in the range [0,1]   *
    *  into a 3x3 rotation matrix, M.  Uniformly distributed random variables *
    *  x0, x1, and x2 create uniformly distributed random rotation matrices.  *
    *  To create small uniformly distributed "perturbations", supply          *
    *  samples in the following ranges                                        *
    *                                                                         *
    *      theta: x[0] in [ 0, d ]                                                   *
    *      phi  : x[1] in [ 0, 1 ]                                                   *
    *      z    : x[2] in [ 0, d ]                                                   *
    *                                                                         *
    * where 0 < d < 1 controls the size of the perturbation.  Any of the      *
    * random variables may be stratified (or "jittered") for a slightly more  *
    * even distribution.
    I add the amplitude (d) as argument
    """
    if randgen is None:
        randgen = np.random.RandomState()

    theta, phi, z = tuple(randgen.rand(3).tolist())
    theta = theta * 2.0*np.pi * amplitude # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0 #* amplitude # For magnitude of pole deflection.

    print(f'theta is {np.rad2deg(theta)}')

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = ( np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z)) #

    st = np.sin(theta)
    ct = np.cos(theta)
    Sx = Vx * ct - Vy * st;
    Sy = Vx * st + Vy * ct;

    # Construct the rotation matrix  ( V Transpose(V) - I ) R, which is equivalent to V S - R.                                        */
    aff = np.eye(4, 4);
    aff[0,0] = Vx * Sx - ct;
    aff[0,1] = Vx * Sy - st;
    aff[0,2] = Vx * Vz;

    aff[1,0] = Vy * Sx + st;
    aff[1,1] = Vy * Sy - ct;
    aff[1,2] = Vy * Vz;

    aff[2,0] = Vz * Sx;
    aff[2,1] = Vz * Sy;
    aff[2,2] = 1.0 - z;   # This equals Vz * Vz - 1.0

    #same ...
    #R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    #M = (np.outer(V, V) - np.eye(3)).dot(R)
    return aff


np.random.seed(4)
nb_mean=50
euler_mean, euler_choral, euler_exp, euler_pol, euler_slerp, euler_qr_slerp = np.zeros((nb_mean,3)), np.zeros((nb_mean,3)), np.zeros((nb_mean,3)), np.zeros((nb_mean,3)), np.zeros((nb_mean,3)), np.zeros((nb_mean,3))
for i in range(nb_mean):
    #rot_euler = np.random.normal(size=(10, 3),loc=20,scale=5) #in degree
    rot_euler = np.random.uniform(-20,20,size=(10, 3)) #in degree
    #print(f'min {np.min(rot_euler)} max {np.max(rot_euler)}')
    #aff_list = get_affine_rot_from_euler(rot_euler) #4*4 affine matrix
    aff_list = [spm_matrix(np.hstack([[0,0,0], rot, [1,1,1,0,0,0]])) for rot in rot_euler]
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

#print(f'max diff euler between dq slerp and euler mean {np.max(np.abs(euler_slerp - euler_mean))}')

print(f'max diff euler between choral and euler mean {np.max(np.abs(euler_choral - euler_mean))}')
print(f'max diff euler between slerp and euler mean {np.max(np.abs(euler_qr_slerp - euler_mean))}')

print(f'max diff euler between slerp and choral  {np.max(np.abs(euler_qr_slerp - euler_choral))}')
print(f'max diff euler between slerp and exp  {np.max(np.abs(euler_qr_slerp - euler_exp))}')
print(f'max diff euler between slerp and polar  {np.max(np.abs(euler_qr_slerp - euler_pol))}')

print(f'max diff euler between exp matrix and choral {np.max(np.abs(euler_exp - euler_choral))}')
print(f'max diff euler between choral and polar {np.max(np.abs(euler_choral - euler_pol))}')




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
