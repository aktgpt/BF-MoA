# BF-MoA

In this repository we provide the code base that accompanies the paper:

## Is brightfield all you need for mechanism of action prediction?

#### by Harrison, Gupta, Rietdijk et al. 

The content and structure of the repo is given by the following: 

```sh
.
├── README.md
├── analysis            : .ipynb files for analysing the results of the trained models and producing the figures included in the paper
├── configs             : .json files for running the models to predict the mechanism of action
├── data                : python scripts for loading the datasets
├── models              : python scripts for the ResNet architecture
├── preprocessing       : .json files for running the models to predict the mechanism of action
├── test                : scripts requied when running the models in test mode
├── train               : scripts requied when running the models in train mode
├── utils               : utility functions                     
└── stats               : .csv files for image statistics and train/val/test splits 
    
```

## preprocessing
The data used in the paper is provided on figshare (LINK TO BE ADDED). The data is for U20S cells treated with 231 compounds belonging to ten different mechanism of action (MoA) classes. Each compound treatment was replicated 6 times, in 6 wells. In each well 5 sites/fields of view were imaged. The tiff images are 16-bit with a 20X magnification and are 2160 x 2160 pixels. For the fluorescence data there are 5 channels and for the brightfiled there a 6 z-planes. The CellProfiler features derived from the fluorescence images are also provided. As the compound treatements were spread across multiple plates we also provide the plate-level DMSO means and standard deviations for normalizing the images. 

On figshare the tiff images were shared in separate folders for each plate, and each image gives only a single channel or z-plane. The "make_numpy.py" script reorganises the data for the models and saves them as multi-channel/multi-z-plane numpy arrays required for the models. The additional files in the "preprocessing" folder are included for completeness. Our entire dataset covered more compounds and MoAs than those used in our paper. The jupyter notebook "bf_split_data.ipynb" was used to create the 5 train/val/test splits for the brightfield data, subsequently "fl_split_data.ipynb" applied the same splits to the fluoresence data. Then "add_dmso_to_splits.py" was used to add an appropriately sized number of the DMSO wells from each plate to the dataset. (All of this selected data is included on figshare). The "make_dmso_stats.py" script was used for computing the DMSO means and standard deviations from all the DMSO data on each plate from our complete dataset.

## model training
To start training execute, for instance, for the brightfield data:
```sh
python main.py -c configs/bf.json -d $TMPDIR/data -r 42
```

This will run the model in training mode and at the end in test mode. A lot of information is automatically saved, including the training and validation loss and accuracy at every iteration and test level statistics at the end. UMAP plots are also produced at the end.

## analysis
In addition to the output generated during and at the end of the model training, in the "analysis" folder we provided jupyter notebooks for recreating the figures and tables from our paper. These include compound-level accuracy comparisons, confusion matrices and classification reports, and our grit score based analysis.
