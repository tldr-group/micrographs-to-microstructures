### Welcome to SliceGAN ###
####### Steve Kench #######
'''
Use this file to define your settings for a training run, or
to generate a synthetic image using a trained generator.
'''

# from slicegan import model, networks, util
# import argparse
import requests  # to get image from the web
from bs4 import BeautifulSoup


def find_and_print_text(soup_block, header):
    """
    :param soup_block: The beautifulsoup object
    :param header: The header for the text to be returned
    :return: (Boolean, text) where the Boolean is true iff the header exists
    in the soup block, and the appropriate text if it exist.
    """
    if soup_block.find(text=header):
        if soup_block.findNext().find(text=header):
            print(f"{header}: {block.findNext('dd').text}")


for i in range(1, 21):
    # Get the url:
    record_url = f'https://www.doitpoms.ac.uk/miclib/full_record.php?id={i}'
    record_request = requests.get(record_url)
    soup = BeautifulSoup(record_request.content, 'html.parser')
    soup_info = soup.find('div', class_='col-md-8')
    soup_info = soup_info.findAll()
    print(f'Micrograph number: {i}')
    for block in soup_info:
        find_and_print_text(block, 'Brief description')
        find_and_print_text(block, 'Further information')
    print()

# Define project name
Project_name = 'NMC_exemplar_final'
# Specify project folder.
Project_dir = 'Trained_Generators/NMC'
# Run with False to show an image during or after training
# parser = argparse.ArgumentParser()
# parser.add_argument('training', type=int)
# args = parser.parse_args()
# Training = args.training
# Project_path = util.mkdr(Project_name, Project_dir, Training)

## Data Processing
# Define image  type (colour, grayscale, three-phase or two-phase.
# n-phase materials must be segmented)
image_type = 'threephase'
# define data type (for colour/grayscale images, must be 'colour' / '
# greyscale. nphase can be, 'tif', 'png', 'jpg','array')
data_type = 'tif'
# Path to your data. One string for isotrpic, 3 for anisotropic
data_path = ['Examples/NMC.tif']

## Network Architectures
# Training image size, no. channels and scale factor vs raw data
img_size, img_channels, scale_factor = 64, 3,  1
# z vector depth
z_channels = 16
# Layers in G and D
lays = 6
# kernals for each layer
dk, gk = [4]*lays, [4]*lays
# strides
ds, gs = [2]*lays, [2]*lays
# no. filters
df, gf = [img_channels,64,128,256,512,1], [z_channels,512,256,128,64,img_channels]
# paddings
dp, gp = [1,1,1,1,0],[2,2,2,2,3]

## Create Networks
# netD, netG = networks.slicegan_nets(Project_path, Training, image_type, dk, ds, df,dp, gk ,gs, gf, gp)

# Train
# if Training:
#     model.train(Project_path, image_type, data_type, data_path, netD, netG, img_channels, img_size, z_channels, scale_factor)
# else:
#     img, raw, netG = util.test_img(Project_path, image_type, netG(), z_channels, lf=6, periodic=[0, 1, 1])
