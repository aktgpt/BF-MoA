# BF-MoA: Mechanism of Action Prediction from Brightfield Images

In this repository we provide the code base that accompanies the BioRxiv paper:

https://doi.org/10.1101/2022.10.12.511869

This paper is also accepted as a poster presentation at 2022 NeurIPS conference workshop on  Learning Meaningful Representations of Life ([LMRL](https://www.lmrl.org/)).


## Is brightfield all you need for mechanism of action prediction?

#### by Ankit Gupta, Philip J Harrison, Håkan Wieslander, Jonne Rietdijk, Jordi Carreras Puigvert, Polina Georgiev, Carolina Wählby, Ola Spjuth, Ida-Maria Sintorn



<p>
    <img src="readme_images/BF-MOA figures.png" alt="drawing" style="width:1200px;"/>
    <center>BF and FL images, activation heatmaps and radar plots for two compounds that were considerably better predicted by BF than both FL and CP models. The BF images show an overlay of the 6 z-planes. The FL images show a merge of the 5 channels with nuclei in blue, ER in cyan, RNA in grey, Golgi/F-actin in green and mitochondria in red. In the heatmaps the outlines of the nuclei and cells from CP are provided. The scale bars in the images represent 20 uM. The radar plots show the affected morphological features according to the CP data.
</center>
</p>





The content and structure of the repo is given by the following: 

```sh
.
├── analysis            : .ipynb files for analysing the results of the trained models and producing the figures included in the paper
├── cellprofiler        : .ipynb file for running the models based on the CellProfiler features
├── configs             : .json files for running the brightfiled and fluorescence models to predict the mechanism of action
├── data                : python scripts for loading the datasets
├── models              : python scripts for the ResNet architecture
├── preprocessing       : scripts for converting the images to numpy, splitting the data and computing normalisation statistics
├── stats               : .csv files for image statistics and train/val/test splits
├── test                : scripts requied when running the models in test mode
├── train               : scripts requied when running the models in train mode              
└── utlis               : utility functions
    
```

## preprocessing
The data used in the paper is provided on figshare (LINK TO BE ADDED). The data is for U20S cells treated with 231 compounds belonging to ten different mechanism of action (MoA) classes. Each compound treatment was replicated 6 times, in 6 wells. In each well 5 sites/fields of view were imaged. The tiff images are 16-bit with a 20X magnification and are 2160 x 2160 pixels. For the fluorescence data there are 5 channels and for the brightfiled there a 6 z-planes. The CellProfiler features derived from the fluorescence images are also provided. As the compound treatements were spread across multiple plates we also provide the plate-level DMSO means and standard deviations for normalizing the images. 

On figshare the tiff images were shared in separate folders for each plate, and each image gives only a single channel or z-plane. The "make_numpy.py" script reorganises the data for the models and saves them as multi-channel/multi-z-plane numpy arrays required for the models. The additional files in the "preprocessing" folder are included for completeness. Our entire dataset covered more compounds and MoAs than those used in our paper. The jupyter notebook "bf_split_data.ipynb" was used to create the 5 train/val/test splits for the brightfield data, subsequently "fl_split_data.ipynb" applied the same splits to the fluoresence data. Then "add_dmso_to_splits.py" was used to add an appropriately sized number of the DMSO wells from each plate to the dataset. (All of this selected data is included on figshare). The "make_dmso_stats.py" script was used for computing the DMSO means and standard deviations from all the DMSO data on each plate from our complete dataset.

## model training
To start training execute, for instance, for the brightfield data:
```sh
python main.py -c configs/bf_non_grit.json -d $TMPDIR/data -r 42
```

This will run the model in training mode and at the end in test mode. A lot of information is automatically saved, including the training and validation loss and accuracy at every iteration and test level statistics at the end. UMAP plots are also produced at the end.

## analysis
In addition to the output generated during and at the end of the model training, in the "analysis" folder we provided jupyter notebooks for recreating the figures and tables from our paper. These include compound-level accuracy comparisons, confusion matrices and classification reports, and our grit score based analysis.
