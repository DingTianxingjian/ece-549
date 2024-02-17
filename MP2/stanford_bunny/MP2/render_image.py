import numpy as np
from generate_scene import get_ball
import matplotlib.pyplot as plt
from generate_scene_bunny import get_bunny
# specular exponent
k_e = 50

def render(Z, N, A, S, 
           point_light_loc, point_light_strength, 
           directional_light_dirn, directional_light_strength,
           ambient_light, k_e):
  # To render the images you will need the camera parameters, you can assume
  # the following parameters. (cx, cy) denote the center of the image (point
  # where the optical axis intersects with the image, f is the focal length.
  # These parameters along with the depth image will be useful for you to
  # estimate the 3D points on the surface of the sphere for computing the
  # angles between the different directions.
  h, w = A.shape
  cx, cy = w / 2, h /2
  f = 128.


  # Ambient Term
  I = A * ambient_light
  
  # Diffuse Term
  # Convert Z to 3D coordinate
  x = np.arange(w) - cx
  y = np.arange(h) - cy
  X, Y = np.meshgrid(x, y)
  X = X * Z / f
  Y = Y * Z / f
  L = point_light_loc - np.dstack((X, Y, Z))  # Light direction vector from each point to the light source
  L_norm = np.linalg.norm(L, axis=2, keepdims=True)
  L = L / L_norm
  # Calculate the dot product between N and L for the diffuse term
  dot_product = np.sum(N * L, axis=2)
  dot_product_1 = np.clip(dot_product, 0, 1)
  I += dot_product_1 * point_light_strength * A
  # Specular Term
  # Calculate the s_i
  S_i = 2 * dot_product[..., np.newaxis] * N - L
  S_i = S_i / np.linalg.norm(S_i, axis=2, keepdims=True)
  V = np.array([0, 0, -1])
  specular_dot = np.sum(S_i * V, axis=2)
  specular_dot = np.clip(specular_dot, 0, 1)
  Is = (specular_dot ** k_e) * point_light_strength * S
  I += Is

  # Calculate the direct light term
  L_direct = directional_light_dirn
  dot_N_L_direct = np.sum(N * L_direct, axis=2)
  I += np.clip(dot_N_L_direct, 0, 1) * directional_light_strength * A

  # Calculate the direct light specluar term
  R_direct = 2 * dot_N_L_direct[..., np.newaxis] * N - L_direct
  R_direct /= np.linalg.norm(R_direct, axis=2, keepdims=True)
  I += (np.clip(np.sum(R_direct * V, axis=2), 0, 1) ** k_e) * directional_light_strength * S

  I = np.minimum(I, 1)*255
  I = I.astype(np.uint8)
  I = np.repeat(I[:,:,np.newaxis], 3, axis=2)
  return I

def main():
  for specular in [True, False]:
    # get_ball function returns:
    # - Z (depth image: distance to scene point from camera center, along the
    # Z-axis)
    # - N is the per pixel surface normals (N[:,:,0] component along X-axis
    # (pointing right), N[:,:,1] component along Y-axis (pointing down),
    # N[:,:,2] component along Z-axis (pointing into the scene)),
    # - A is the per pixel ambient and diffuse reflection coefficient per pixel,
    # - S is the per pixel specular reflection coefficient.
    Z, N, A, S = get_bunny(specular=specular)

    # Strength of the ambient light.
    ambient_light = 0.5
    
    # For the following code, you can assume that the point sources are located
    # at point_light_loc and have a strength of point_light_strength. For the
    # directional light sources, you can assume that the light is coming _from_
    # the direction indicated by directional_light_dirn (\hat{v}_i = directional_light_dirn), and with strength
    # directional_light_strength. The coordinate frame is centered at the
    # camera, X axis points to the right, Y-axis point down, and Z-axis points
    # into the scene.
    
    # Case I: No directional light, only point light source that moves around
    # the object. 
    point_light_strength = [1.5]
    directional_light_dirn = [[1, 0, 0]]
    directional_light_strength = [0.0]
    
    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16): 
      point_light_loc = [[10*np.cos(theta), 10*np.sin(theta), -3]]
      I = render(Z, N, A, S, point_light_loc, point_light_strength, 
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e)
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_point.png', bbox_inches='tight')
    plt.close()

    # Case II: No point source, just a directional light source that moves
    # around the object.
    point_light_loc = [[0, -10, 2]]
    point_light_strength = [0.0]
    directional_light_strength = [2.5]
    
    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16): 
      directional_light_dirn = [np.array([np.cos(theta), np.sin(theta), .1])]
      directional_light_dirn[0] = \
        directional_light_dirn[0] / np.linalg.norm(directional_light_dirn[0])
      I = render(Z, N, A, S, point_light_loc, point_light_strength, 
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e) 
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_direction.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
  main()
