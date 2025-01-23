# Generative Networks for Spatio-Temporal Gap Filling of Sentinel-2 Reflectances
---------------
| [Journal ISPRS paper](https://doi.org/10.1016/j.isprsjprs.2025.01.016) |

# Abstract

Earth observation from satellite sensors offers the possibility to monitor natural ecosystems by deriving spatially explicit and temporally resolved biogeophysical parameters. Optical remote sensing, however, suffers from missing data mainly due to the presence of clouds, sensor malfunctioning, and atmospheric conditions. This study proposes a novel deep learning architecture to address gap filling of satellite reflectances, more precisely the visible and near-infrared bands, and illustrates its performance at high-resolution Sentinel-2 data. We introduce GANFilling, a generative adversarial network capable of sequence-to-sequence translation,  which comprises convolutional long short-term memory layers to effectively exploit complete dependencies in space-time series data. We focus on Europe and evaluate the method's performance quantitatively (through distortion and perceptual metrics) and qualitatively (via visual inspection and visual quality metrics). Quantitatively, our model offers the best trade-off between denoising corrupted data and preserving noise-free information, underscoring the importance of considering multiple metrics jointly when assessing gap filling tasks. Qualitatively, it successfully deals with various noise sources, such as clouds and missing data, constituting a robust solution to multiple scenarios and settings. We also illustrate and quantify the quality of the generated product in the relevant downstream application of vegetation greenness forecasting, where using GANFilling enhances forecasting in approximately 70% of the considered regions in Europe. This research contributes to underlining the utility of deep learning for Earth observation data, which allows for improved spatially and temporally resolved monitoring of the Earth surface.

# Usage and Requirements

## Train

`
python src/train.py --dataPath data/train_data --cleanDataPath src/utils/example_clean_data.json --name model_1
`

Models are saved to ./trained_models/ (can be changed by passing --modelsPath=your_dir in train.py). 

For training the discriminator, a json file with the noiseless(real) data has to be generated. This can be done with the ./utils/generate_cleanData_file.py. An example of the structure of this file can be found in ./utils/generate_cleanData_file.py.  

The folder with the training data is specified with the `--dataPath` argument. This repository just includes few samples in data/train_data/ as a reference for structure and behaviour checking. 

## Test 

`
python src/test.py --model_path trained_models/GANFilling.pt --data_path data --results_path results
`

This will run the GANFilling trained model on a small set of examples and generate the corresponding gap filled time series. 

To test your own model modify the parameter `--model_path` in test.py.
To test on your own data modify the parameter `--data_path` in test.py.

# Results

![image](GANFilling_percept_results.png)

<b>Example images showing the GANFilling reconstruction on different land covers.</b> For each example, the first row shows the land cover map followed by ten original time steps of a visible (RGB) sequence, while the second row corresponds to its noise-free version. All images are noted with the day-of-year (DOY). The third row illustrates the NDVI maps for the noise-free images. Different types of noise are outlined in red. (A) Complexe scene with predominant herbaceous vegetation and multiple frames with complete loss of information. (B) Sequence with mostly cultivated areas to show the performance on fast changes in the Earth’s surface with heavily occluded frames. (C) Sequence with widespread vines characterized by a rapid evolution of the land cover. (D) Predominant coniferous tree cover with a water body nearby. (E) Sequence with predominant broadleaf tree cover and several consecutive time steps with dense occlusions. Land cover’s legend: <span style="color:rgb(255,255,255)">&#9723;</span> Cloud or No data, <span style="color:rgb(210,0,0)">&#9724;</span> Artificial surfaces and constructions, <span style="color:rgb(253,211,39)">&#9724;</span> Cultivated areas, <span style="color:rgb(176,91,16)">&#9724;</span>  Vineyards, <span style="color:rgb(35,152,0)">&#9724;</span>  Broadleaf tree cover, <span style="color:rgb(8,98,0)">&#9724;</span>  Coniferous tree cover, <span style="color:rgb(249,150,39)">&#9724;</span>  Herbaceous vegetation, <span style="color:rgb(141,139,0)">&#9724;</span>  Moors and Heathland, <span style="color:rgb(95,53,6)">&#9724;</span>  Sclerophyllous vegetation, <span style="color:rgb(149,107,196)">&#9724;</span>  Marshes, <span style="color:rgb(77,37,106)">&#9724;</span>  Peatbogs, <span style="color:rgb(154,154,154)">&#9724;</span>  Natural material surfaces, <span style="color:rgb(106,255,255)">&#9724;</span>  Permanent snow covered surfaces, <span style="color:rgb(20,69,249)">&#9724;</span>  Water bodies.

# How to cite

If you use this code for your research, please cite our paper Generative Networks for Spatio-Temporal Gap Filling of Sentinel-2 Reflectances:

```
@article{GonzalezCalabuig2025,
title = {Generative networks for spatio-temporal gap filling of Sentinel-2 reflectances},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {220},
pages = {637-648},
year = {2025},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.01.016},
author = {Maria Gonzalez-Calabuig and Miguel-Ángel Fernández-Torres and Gustau Camps-Valls}}
```

# Aknowledgments

Authors acknowledge the support from the European Research Council (ERC) under the ERC SynergyGrant USMILE (grant agreement 855187), the European Union’s Horizon 2020 research and innovation program within the projects ‘XAIDA: Extreme Events- Artificial Intelligence for Detection and Attribution,’(grant agreement 101003469), ‘DeepCube: Explainable AI pipelines for big Copernicus data’ (grant agreement 101004188), the ESA AI4Science project ”MultiHazards, Compounds and Cascade events: DeepExtremes”, 2022-2024, the computer resources provided by the J¨ulich Supercomputing Centre (JSC) (Project No.PRACE-DEV-2022D01-048), the computer resources provided by Artemisa (funded by the European Union ERDF and Comunitat Valenciana), as well as the technical support provided by the Instituto de Física Corpuscular, IFIC (CSIC-UV).

# License

[MIT]()