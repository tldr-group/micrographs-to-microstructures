from slicegan import networks, util
import argparse
import torch
import tifffile
import numpy as np
from plotoptix.materials import make_material
import plotoptix.materials as m
from plotoptix import NpOptiX, TkOptiX
from scipy import ndimage
import moviepy.editor as mp
from moviepy.editor import *

import imageio
import os
from time import time

start = time()
# Define project name
parser = argparse.ArgumentParser()
parser.add_argument('micro', type=str)
args = parser.parse_args()

Project_name = f'microstructure{args.micro}'
# Specify project folder.
Project_dir = 'Trained_Generators'
# Run with False to show an image during or after training
parser = argparse.ArgumentParser()
Project_path = util.mkdr(Project_name, Project_dir, False)

os.makedirs(Project_path + 'frames', exist_ok=True)

image_type = 'grayscale'
data_type = 'png'
data_path = ['Examples/micro378.png']

img_size, img_channels, scale_factor = 64, 1, 1
z_channels = 16
lays = 6
dk, gk = [4] * lays, [4] * lays
ds, gs = [2] * lays, [2] * lays
df, gf = [img_channels, 64, 128, 256, 512, 1], [z_channels, 512, 256, 128, 64,
                                                img_channels]
dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

## Create Networks
netD, netG = networks.slicegan_nets(Project_path, False, image_type, dk, ds,
                                    df, dp, gk, gs, gf, gp)
netG = netG()
netG.eval()
lf = 8
n = (lf - 2) * 32
noise = torch.randn(1, z_channels, lf, lf, lf)
netG = netG.cuda()
noise = noise.cuda()
nseeds = 10
blur = 0.75
interval_size = 360 / nseeds
netG.load_state_dict(torch.load(Project_path + '_Gen.pt'))
res = 512
min_accum = 500
img = netG(noise[0].unsqueeze(0))
image_type = 'twophase' if img.shape[1] == 2 else 'grayscale'
print(img.shape, image_type)
img = util.post_proc(img, image_type)
tif = np.int_(img)
tifffile.imsave(Project_path + ".tif", tif)

if image_type == 'twophase':
    ph = 0 if np.mean(img) > 0.5 else 1
if image_type == 'twophase':
    bind = np.array(np.where(img == ph)).T
    c = 1 - (bind + 0.5)
    c = (0.3, 0.3, 0.3)
else:
    # img = ndimage.gaussian_filter(img, blur)
    bind = np.array(np.where(img != -1)).T
    # bind[:,1][img.reshape(-1) <  0] +=1000
    c = (img.reshape(-1)) / 255
    print(c.max(), c.min(), c.shape)
bind = bind / n - 0.5


class params():
    tf = 360
    f = 0
    s = int(360 / tf)
    e = [-3, 1.2, 0]
    l = [-5, 10, 0]
    bind = bind
    c = c
    fin = False


def compute(rt: NpOptiX,
            delta: int) -> None:  # compute scene updates in parallel to the raytracing
    interval = int(params.f // interval_size)
    # print(interval, params.f)
    # noise1 = noise[interval]
    # try:
    #     noise2 = noise[interval+1]
    # except:
    #     noise2 = noise[0]
    # prog = (params.f%interval_size)/interval_size

    # noiset = (noise1 * (1-prog)) + (noise2*prog)
    # img = util.post_proc(netG(noise), image_type)
    # if image_type == 'twophase':
    #     # bind = np.array(np.where(img == ph)).T
    #     # params.c = 1 -(bind + 0.5)
    #     params.c = (0.3,0.3,0.3)
    # else:
    #    img = ndimage.gaussian_filter(img, blur)
    #    bind = np.array(np.where(img !=-1)).T
    # cutoff = params.f/180 if params.f < 180 else 2 - params.f/180
    # bind[:,1][img.reshape(-1) < img.max() * cutoff * 0.5] +=1000
    # params.c = img.reshape(-1)/100

    # params.bind = bind/n - 0.5
    params.f += params.s
    f_step = params.f * np.pi * 2 / 360
    # params.e = [0.5*np.cos(params.f/360), 12, 20*np.sin(params.f/360)]
    x, y = np.cos(f_step), np.sin(f_step)

    params.e = [-3 * x, 1.2, -3 * y]
    params.l = [-5 * x, 5, -5 * y]
    # print(params.e)


# optionally, save every frame to a separate file using save_image() method
def update(rt: NpOptiX) -> None:
    rt.update_camera('cam1', eye=params.e)
    # rt.update_light('light1', pos=params.l)

    rt.set_data("cubes_b", pos=params.bind, u=[1 / n, 0, 0], v=[0, 1 / n, 0],
                w=[0, 0, 1 / n],
                geom="Parallelepipeds",  # cubes, actually default geometry
                mat="diffuse",  # opaque, mat, default
                c=params.c)
    # rt.update_light('light1', pos=params.l)
    # print("frames/frame_{:05d}.png".format(params.f))
    rt.save_image(Project_path + "frames/frame_{:05d}.png".format(params.f))
    if params.f % nseeds == 0:
        print(params.f, time() - start)

    if params.f == 360:
        save_animation()
        rt.close()
        params.fin = True


def animate(netG):
    width = res;
    height = res
    optix = NpOptiX(on_scene_compute=compute,
                    on_rt_completed=update,
                    width=width, height=height,
                    start_now=False)

    optix.set_param(min_accumulation_step=min_accum,
                    # 1 animation frame = 128 accumulation frames
                    max_accumulation_frames=5130,
                    light_shading="Hard")  # accumulate 512 frames when paused
    optix.set_uint("path_seg_range", 5, 10)
    exposure = 1;
    gamma = 2.3
    optix.set_float("tonemap_exposure", exposure)  # sRGB tonning
    optix.set_float("tonemap_gamma", gamma)
    optix.set_float("denoiser_blend", 0.25)
    optix.add_postproc("Denoiser")
    optix.set_background(250)
    # img = util.post_proc(netG(noise[0].unsqueeze(0)), image_type)
    # if image_type == 'twophase':

    #     bind = np.array(np.where(img == ph)).T
    #     c = 1 -(bind + 0.5)
    #     c = (0.3,0.3,0.3)
    # else:
    #     # img = ndimage.gaussian_filter(img, blur)
    #     bind = np.array(np.where(img !=-1)).T
    #     # bind[:,1][img.reshape(-1) <  0] +=1000
    #     c = (img.reshape(-1))/100
    n = img.shape[0]
    s = 1 / n
    alpha = np.full((1, 1, 4), 1, dtype=np.float32)
    optix.set_texture_2d("mask", (255 * alpha).astype(np.uint8))
    m_diffuse_3 = make_material("Diffuse", color_tex="mask")
    optix.setup_material("3", m_diffuse_3)
    optix.set_data("cubes_b", pos=bind, u=[s, 0, 0], v=[0, s, 0], w=[0, 0, s],
                   geom="Parallelepipeds",  # cubes, actually default geometry
                   mat="3",  # opaque, mat, default
                   c=c)
    optix.setup_camera("cam1", eye=params.e, target=[0, 0, 0], up=[0, 1, 0],
                       fov=30)
    optix.set_ambient((0.3, 0.3, 0.3))
    x = n / 2
    optix.start()


def save_animation():
    frames = sorted(os.listdir(Project_path + 'frames'))
    print('loading frames')
    fps = 45
    frame_duration = 1 / fps
    clips = [
        ImageClip(f'{Project_path}frames/{m}').set_duration(frame_duration)
        for m in frames]
    clip = concatenate_videoclips(clips, method="compose")
    clip.write_videofile(f'{Project_path}.mp4', fps=fps)


animate(netG)
# save_animation()
print('finished', time() - start)

