
##
# Jemely Robles
# cs 73, Assignment 4
##

import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from numpy.linalg import svd
from cp_hw4 import integrate_poisson, integrate_frankot, load_sources
from matplotlib.colors import LightSource

##
#########################
# Initials
#########################
##
# Initialize a list to store the luminance channels
luminance_channels = [
    color.rgb2xyz(io.imread(f"./data/input_{i}.tif")[:, :, :3])[:, :, 1].flatten()
    for i in range(1, 8)
]

# Stack the luminance channels into a matrix
I = np.vstack(luminance_channels)

print("stack working")


##
#########################
# Uncalibrated photometric stereo
#########################
##
image = io.imread("./data/input_1.tif")
height, width = image.shape[:2]

# Perform SVD on matrix I
Le, S, Be = np.linalg.svd(I, full_matrices=False)

sqrt_S = np.sqrt(S)
Le = Le[:, :3] * sqrt_S[:3]
Be = Be[:3, :] * sqrt_S[:3, np.newaxis]


# Compute albedoes (Ae) and normals (Ne)
Ae = np.linalg.norm(Be, axis=0)
Ne = Be / Ae


reshape_func = lambda x: x.reshape(image.shape[:2])
Ae_image = Ae.reshape((height, width))
Ne_image = Ne.T.reshape((height, width, 3))

plt.imshow(Ae_image, cmap='gray')
plt.title('Albedo')
plt.colorbar()
plt.show()

plt.imshow((Ne_image + 1)/2)
plt.title('Normals')
plt.axis('off')
plt.show()
print("photometric stereo working")

theta = np.pi/3
Q = np.matrix(((np.cos(theta), -np.sin(theta), 0), (np.sin(theta), np.cos(theta), 0), (0, 0, 1)))

LQ = Le @ Q
BQ = np.linalg.inv(Q).T * Be
AQ = np.linalg.norm(BQ, axis=0)
NQ = BQ / AQ

reshape_func = lambda x: x.reshape(image.shape[:2])
AQ_image = reshape_func(AQ)
NQ_image = np.transpose(np.array(list(map(reshape_func, NQ))), (1, 2, 0))

plt.imshow(AQ_image, cmap='gray')
plt.title('Albedo, Q')
plt.colorbar()
plt.show()

plt.imshow((NQ_image + 1)/2)
plt.title('Normals, Q')
plt.axis('off')
plt.show()


##
#########################
# Enforcing  integrability
#########################
##

# Construct the homogeneous linear system A·x=0
I_B = Be.T.reshape((height, width, 3))
print("hi")

blur_sigma = 10

blurred_B = np.dstack(
    (gaussian_filter(I_B[:, :, 0], blur_sigma), 
     gaussian_filter(I_B[:, :, 1], blur_sigma), 
     gaussian_filter(I_B[:, :, 2], blur_sigma)))
print("3")

# find the gradients of the blurred image
grad_y, grad_x, grad_z = np.gradient(blurred_B)

grad_y = grad_y.reshape((height * width, 3))
grad_x = grad_x.reshape((height * width, 3))

# create the linear system of equations
A = np.zeros((height * width, 6))

A[:, 0] = (Be.T[:, 0] * grad_x[:, 1]) - (Be.T[:, 1] * grad_x[:, 0])
A[:, 1] = (Be.T[:, 0] * grad_x[:, 2]) - (Be.T[:, 2] * grad_x[:, 0])
A[:, 2] = (Be.T[:, 1] * grad_x[:, 2]) - (Be.T[:, 2] * grad_x[:, 1])
A[:, 3] = -(Be.T[:, 0] * grad_y[:, 1]) + (Be.T[:, 1] * grad_y[:, 0])
A[:, 4] = -(Be.T[:, 0] * grad_y[:, 2]) + (Be.T[:, 2] * grad_y[:, 0])
A[:, 5] = -(Be.T[:, 1] * grad_y[:, 2]) + (Be.T[:, 2] * grad_y[:, 1])

G = np.matrix([[1, 0, 0],
               [0, 1, 0], 
               [0, 0, -1]])

U, s, Vh = svd(A, full_matrices=False)

x = Vh[-1, :]
delta = np.matrix([[-x[2], x[5], 1], 
                   [x[1], -x[4], 0],
                   [-x[0], x[3], 0]])


Be_updated = np.matmul(np.linalg.inv(delta), Be)


Be_updated = np.asarray(Be_updated)


Be_updated_image = np.transpose(np.array(list(map(reshape_func, Be_updated))), (1, 2, 0))



A_pri = np.sqrt(np.sum(Be_updated**2, axis=0))
N_pri = Be_updated/A_pri

IA_pri = np.reshape(A_pri, (height,width))
IN_pri = np.reshape(N_pri.T, (height,width,3))
IB_pri = np.reshape(Be_updated.T, (height,width,3))


plt.imshow((IA_pri), cmap='gray')
plt.title('Albedoes after enforcing integrability')
plt.show()
plt.imshow((IN_pri+1)/2)
plt.title('Normal image after enforcing integrability')
plt.show()


##
#########################
# Normal integration
#########################
##
zx = IN_pri[:,:,0]/IN_pri[:,:,2]
zy = IN_pri[:,:,1]/IN_pri[:,:,2]
Z_poisson = integrate_poisson(zx, zy)

Z_frankot = integrate_frankot(zx, zy)

# Visualize the depth image
plt.imshow(Z_poisson, cmap='gray')
plt.title('Depth Image (Poisson)')
plt.colorbar()
plt.show()

ls = LightSource()
color_shade = ls.shade(-Z_poisson, plt.cm.gray)
x, y = np.meshgrid(np.arange(width), np.arange(height))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, -Z_poisson, facecolors=color_shade, rstride=4, cstride=4)
ax.set_title('3D Surface (Poisson)')
plt.show()

plt.imshow(Z_frankot, cmap='gray')
plt.title('Depth Image (Frankot)')
plt.colorbar()
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, -Z_frankot, facecolors=color_shade, rstride=4, cstride=4)
ax.set_title('3D Surface (frankot)')
plt.show()


##
#########################
# Calibrated  photometric  stereo
#########################
##

L = load_sources()
print("L shape", L.shape)
print("I shape ", I.shape)

B_recovered = np.linalg.pinv(L) @ I 

# Compute albedoes (Ae) and normals (Ne)
Ae_recovered = np.linalg.norm(B_recovered, axis=0)
Ne_recovered = B_recovered / Ae_recovered

Ae_recovered_image = reshape_func(Ae_recovered)
plt.imshow(Ae_recovered_image, cmap='gray')
plt.title('Recovered Albedoes')
plt.colorbar()
plt.show()

Ne_recovered_image = np.transpose(np.array(list(map(reshape_func, Ne_recovered))), (1, 2, 0))
plt.imshow((Ne_recovered_image + 1) / 2)
plt.title('Recovered Normals')
plt.axis('off')
plt.show()

Z_recovered_frankot = integrate_frankot(Ne_recovered_image[:, :, 0], Ne_recovered_image[:, :, 1])

plt.imshow(Z_recovered_frankot, cmap='gray')
plt.title('Recovered Depth Image (Frankot)')
plt.show()

