
This repository is for the paper [FusionESP: Improved enzyme-substrate pair prediction by fusing protein and chemical knowledge](https://www.biorxiv.org/content/10.1101/2024.08.13.607829v2)

The web server is accessible at https://rqkjkgpsyu.us-east-1.awsapprunner.com/

## implementation the protocol locally through terminal
We have tested several different combination in our manuscript. The following example is to implement the FusionESP with ESM-2-2560	+ MoLFormer, the architecture with the best perforamnce in our experiments. 
It is worth noting that our model architecture can also be directly applied for drug-target interaction prediction, protien compound interaction, etc. Feel free to have a try.

### Environment 
Python 3.9.18, based on Mac OS with Apple M2 chip 
```
pip install -r requirements.txt
```
//// we also test the code at GoogleColab with Python 3.10.12 and default package 
'''
torch==2.5.1+cu121
fair-esm==2.0.0
rdkit==2024.03.6
'''
### Dataset 

The original dataset are also uploaded to [Zenodo](https://zenodo.org/records/13891018).The dataset was downloaded from the our references without any modification. The dataset was divided into training, validation, and test sets as in the original paper. [paper 1](https://www.nature.com/articles/s41467-023-38347-2) and [paper 2](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012100). They can also be downloaded from Google Drive at save at [GoogleDrive link](https://drive.google.com/drive/folders/1op-L1iG55gGDhSCCXT9z62c9kJhoQ769?usp=drive_link).

### embeddings 
The 
```
!python generate_embeddings.py --filename df_test.csv
```

All the implementation was conducted in Google Colab + Google Drive. The repository provided the original notebook for reproduction. 

The original data, embedding files and models are save at [GoogleDrive link](https://drive.google.com/drive/folders/1op-L1iG55gGDhSCCXT9z62c9kJhoQ769?usp=drive_link)


