import os
import sys
import torch 
import csv
import numpy as np

import argparse
from pathlib import Path
from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append('src/models')
from generator import *


def main(config):

    gap_filling_model_path = config.model_path
    results_path = Path(config.results_path)
    data_path = Path(config.data_path)
    list_samples = [x for x in os.listdir(config.data_path) if x.endswith('.npz')]


    landcover_cmap = {}
    with open('data/landcover_types.csv', 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            landcover_cmap[row[0]] = [row[1], row[2], row[3]]
    
    for sample in list_samples:

        sample_lc = data_path / "landcover" / sample.replace("context_", "")

        landcover = np.load(sample_lc)['landcover'][0]
        context = np.load(data_path / sample)['highresdynamic'][:,:,:4, :10]

        model = __load_gapFill_model(gap_filling_model_path)
        gap_filled_sample = __gapFill_context(context, model)

        
        n = gap_filled_sample[0,-1]
        r = gap_filled_sample[0,2]
        ndvi = (n-r)/(n+r)

        fig, axs = plt.subplots(3, 11, figsize=(15, 4))

        landcover_rgb = np.empty((128,128, 3))   
        for c in np.unique(landcover):
            for ch in range(3):
                landcover_rgb[landcover==c, ch] = landcover_cmap[str(c)][ch]

        landcover_rgb = Image.fromarray(landcover_rgb.astype('uint8'), 'RGB')
        axs[0, 0].imshow(landcover_rgb)
        axs[0, 0].axis('off')
        axs[1, 0].axis('off')
        axs[2, 0].axis('off')

        import datetime
        date = datetime.datetime.strptime(sample.split('_')[2], '%Y-%m-%d')


        for i in range(10):
            img_rgb = np.clip(context[:,:,:3, i]*3,0,1)[:,:,[2,1,0]]            
            img_rgb = (img_rgb * 255).astype('uint8')
            img_rgb = Image.fromarray(img_rgb)
            axs[0, i+1].imshow(img_rgb) 
            axs[0, i+1].text(94, 122, (date+datetime.timedelta(days=5*i)).strftime('%j'), color='black', fontsize=10, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
            axs[0, i+1].axis('off')
        
        for i in range(10):
            img_rgb = np.transpose(gap_filled_sample.cpu().detach().numpy()[0, [2,1,0],:,:, i], (1, 2, 0))
            img_rgb = np.clip(img_rgb*3, 0, 1)
            img_rgb = (img_rgb * 255).astype('uint8')
            img_rgb = Image.fromarray(img_rgb)
            axs[1, i+1].imshow(img_rgb) 
            axs[1, i+1].axis('off')
        
        cmap = mpl.colormaps.get_cmap('jet')
        cmap.set_bad(color='black')
        
        for i in range(10):
            ndvi_img = Image.fromarray(ndvi[:,:,i].cpu().detach().numpy())
            im = axs[2, i+1].imshow(ndvi[:,:,i].cpu().detach().numpy(), cmap=cmap, vmin=0., vmax=1.) 
            axs[2, i+1].axis('off')
        
        cbar_ax = fig.add_axes([0.2, 0.95, 0.7, 0.03])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.outline.set_linewidth(0)

        plt.subplots_adjust(wspace=0., hspace=0.05)

        landcover_cmap_legend = {}
        with open('data/landcover_types.csv', 'r') as fd:
            reader = csv.reader(fd)
            for row in reader:
                landcover_cmap_legend[row[5]] = [int(row[1])/255, int(row[2])/255, int(row[3])/255, 1]
        
        landcover_cmap_legend.pop('No data')
        legend_items =[mpatches.Patch(color=color,label=lc) for lc, color in landcover_cmap_legend.items()]

        path = results_path / sample
        plt.savefig(f"{path}.png", bbox_inches='tight')
        plt.close()


## GAP FILLING FUNCTIONS

def __load_gapFill_model(gapFilling_model):
    modelConfiguration = torch.load(gapFilling_model, map_location='cpu')
    trainConfiguration = modelConfiguration['configuration']
    model = Generator('cpu' , trainConfiguration.inputChannels, trainConfiguration.outputChannels)
    model.load_state_dict(modelConfiguration['model'])
    model.eval()
    return model

def __prepare_data(context):
    context = np.nan_to_num(np.clip(context, 0, 1), nan=1.0)
    context = np.transpose(context, (2,0,1,3))
    return context

def __gapFill_context(context, model):
    context = __prepare_data(context)
    tensor = torch.from_numpy(context).unsqueeze(dim=0).float()
    output, _ = model(tensor)
    return output

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = "ArgParse")
    parser.add_argument('--model_path', type=str) 
    parser.add_argument('--data_path', type=str) 
    parser.add_argument('--results_path', type=str) 
    
    config = parser.parse_args()

    main(config)