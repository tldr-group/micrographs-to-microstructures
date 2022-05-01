from slicegan import networks, util
import argparse
import torch
import numpy as np
from plotoptix.materials import make_material
import plotoptix.materials as m
from plotoptix import NpOptiX, TkOptiX
from scipy import ndimage
import moviepy.editor as mp

# Define project name
parser = argparse.ArgumentParser()
parser.add_argument('micro', type=str)
args = parser.parse_args()

Project_name = f'micro{args.micro}'
# Specify project folder.
Project_dir = 'Trained_Generators/micros'
# Run with False to show an image during or after training
parser = argparse.ArgumentParser()
Project_path = util.mkdr(Project_name, Project_dir, False)

image_type = 'grayscale'
data_type = 'png'
data_path = ['Examples/micro378.png']

img_size, img_channels, scale_factor = 64, 1,  1
z_channels = 16
lays = 6
dk, gk = [4]*lays, [4]*lays
ds, gs = [2]*lays, [2]*lays
df, gf = [img_channels,64,128,256,512,1], [z_channels,512,256,128,64,img_channels]
dp, gp = [1,1,1,1,0],[2,2,2,2,3]

## Create Networks
netD, netG = networks.slicegan_nets(Project_path, False, image_type, dk, ds, df,dp, gk ,gs, gf, gp)
netG = netG()
netG.eval()
lf = 8
n = (lf-2) * 32
noise = torch.randn(10, z_channels, lf, lf, lf)
netG = netG.cuda()
noise = noise.cuda()
nseeds=10
blur = 0.75
interval_size = 360/nseeds
netG.load_state_dict(torch.load(Project_path + '_Gen.pt'))
res = 1920
min_accum = 512
img = netG(noise[0].unsqueeze(0))
image_type = 'twophase' if img.shape[1]==2 else 'grayscale' 
print(img.shape, image_type)
img = util.post_proc(img, image_type)
if image_type =='twophase':
    ph = 0 if np.mean(img) > 0.5 else 1

class params():
    tf = 360
    f = 0
    s = int(360/tf)
    e = [3, 1.2, 0]
    l = [0, 10, 5]
    bind = None
    c = None
def compute(rt: NpOptiX, delta: int) -> None: # compute scene updates in parallel to the raytracing
    interval = int(params.f//interval_size)
    # print(interval, params.f)
    noise1 = noise[interval]
    try:
        noise2 = noise[interval+1]
    except:
        noise2 = noise[0]
    prog = (params.f%interval_size)/interval_size
    # prog = prog if prog <=1 else 2- prog
    # print(prog, interval, params.f)
    noiset = (noise1 * (1-prog)) + (noise2*prog)
    img = util.post_proc(netG(noiset.unsqueeze(0)), image_type)
    if image_type == 'twophase':

        bind = np.array(np.where(img == ph)).T
        params.c = 1 -(bind + 0.5)
        params.c = (0.3,0.3,0.3)
    else:
        img = ndimage.gaussian_filter(img, blur)
        bind = np.array(np.where(img !=-1)).T
        cutoff = params.f/180 if params.f < 180 else 2 - params.f/180
        bind[:,1][img.reshape(-1) < img.max() * cutoff * 0.5] +=1000
        params.c = img.reshape(-1)/255

    params.bind = bind/n - 0.5
    params.f+=params.s
    f_step = params.f * np.pi * 2 / 360
    # params.e = [0.5*np.cos(params.f/360), 12, 20*np.sin(params.f/360)]
    x, y = np.cos(f_step), np.sin(f_step)
    
    params.e = [-3*x, 1.2, -3*y]
    params.l = [-5*x, 5, -5*y]
    # print(params.e)
    
# optionally, save every frame to a separate file using save_image() method
def update(rt: NpOptiX) -> None: 
    rt.update_camera('cam1', eye=params.e)
    # rt.update_light('light1', pos=params.l)

    rt.set_data("cubes_b", pos=params.bind, u=[1/n, 0, 0], v=[0, 1/n, 0], w=[0, 0, 1/n],
                geom="Parallelepipeds", # cubes, actually default geometry
                mat="diffuse",          # opaque, mat, default
                c = params.c)
    # rt.update_light('light1', pos=params.l)
    # print("frames/frame_{:05d}.png".format(params.f))
    rt.save_image("frames/frame_{:05d}.png".format(params.f))    
    if params.f%nseeds ==0:
        print(params.f)
    if params.f==360:
        save_animation()
        rt.close()
        

def animate(netG):
    width = res; height = res
    optix = NpOptiX(on_scene_compute=compute,
                    on_rt_completed=update,
                    width=width, height=height,
                    start_now=False)

    # optix = TkOptiX(start_now=False) # no need to open the window yet
    optix.set_param(min_accumulation_step=min_accum,    # 1 animation frame = 128 accumulation frames
                    max_accumulation_frames=5130,
                    light_shading="Hard")  # accumulate 512 frames when paused
    optix.set_uint("path_seg_range", 5, 10)
    exposure = 0.8; gamma = 2.2
    optix.set_float("tonemap_exposure", exposure) # sRGB tonning
    optix.set_float("tonemap_gamma", gamma)
    optix.set_float("denoiser_blend", 0.25)
    optix.add_postproc("Denoiser")
    # optix.set_ambient([0.1, 0.2, 0.3])
    optix.set_background(255)



    # optix.set_float("tonemap_exposure", 0.5)
    # optix.set_float("tonemap_gamma", 2.2)

    # optix.add_postproc("Gamma") 

    img = util.post_proc(netG(noise[0].unsqueeze(0)), image_type)
    # print(img.shape)
    if image_type == 'twophase':

        bind = np.array(np.where(img == ph)).T
        c = 1 -(bind + 0.5)
        c = (0.3,0.3,0.3)
    else:
        img = ndimage.gaussian_filter(img, blur)
        bind = np.array(np.where(img !=-1)).T
        bind[:,1][img.reshape(-1) <  0] +=1000
        c = (img.reshape(-1))/100
    n = img.shape[0]
    s=1/n
    bind = bind/n - 0.5
    # use "Hard" light shading for the best caustics and "Soft" for fast convergence

    # optix.set_uint("path_seg_range", 15, 30)

    alpha = np.full((1, 1, 4), 1, dtype=np.float32)
    optix.set_texture_2d("mask", (255*alpha).astype(np.uint8))
    m_diffuse_3 = make_material("Diffuse", color_tex="mask")
    # m_specular_1 = make_material("Reflective", specular=0.9)
    # m_glass_4 = make_material("Transmissive", refraction_index=[1.35, 1.4, 1.48])
    # m_metal_1 = make_material("Reflective", metalness=1.0, roughness=0.004)
    optix.setup_material("3", m_diffuse_3)
    # optix.setup_material("1", m_specular_1)
    # optix.setup_material("3", m_metal_1)

    optix.set_data("cubes_b", pos=bind, u=[s, 0, 0], v=[0, s, 0], w=[0, 0, s],
                geom="Parallelepipeds", # cubes, actually default geometry
                mat="3",          # opaque, mat, default
                c = c)
    optix.setup_camera("cam1", eye=params.e, target=[0,0,0], up=[0,1, 0], fov=30)
    # optix.set_background(100)
    optix.set_ambient((0.3, 0.3, 0.3))


    optix.set_float("tonemap_exposure", 0.5)
    optix.set_float("tonemap_gamma", 2.2)

    optix.add_postproc("Gamma")      # apply gamma correction postprocessing stage, or
    x = n/2
    # optix.setup_light("light1", pos=params.l, color=2*np.array([1.0, 1.0, 1.0]), radius=2)
    optix.start()

def save_animation():
    import imageio
    import os
    images = []
    frames = sorted(os.listdir('frames'))
    for filename in frames[1:]:
        images.append(imageio.imread(f'frames/{filename}'))
    print(f'saving animation {Project_name}')
    imageio.mimsave(f'animations/{Project_name}.gif', images, fps=20)
    clip = mp.VideoFileClip(f'animations/{Project_name}.gif')
    clip.write_videofile(f'animations/{Project_name}.mp4', fps=60)
    os.remove(f'animations/{Project_name}.gif')
animate(netG)

