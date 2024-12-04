
This repository is for the paper [FusionESP: Improved enzyme-substrate pair prediction by fusing protein and chemical knowledge](https://www.biorxiv.org/content/10.1101/2024.08.13.607829v2)

The web server is accessible at https://rqkjkgpsyu.us-east-1.awsapprunner.com/

## implementation the protocol locally through terminal
We have tested several different combination in our manuscript. The following example is to implement the FusionESP with ESM-2-2560	+ MoLFormer, the architecture with the best perforamnce in our experiments. 
It is worth noting that our model architecture can also be directly applied for drug-target interaction prediction, protien compound interaction, etc. Feel free to have a try.

### Environment 
Python 3.9.18, based on Mac OS with Apple M2 chip 
```
$ pip install -r requirements.txt
```
//// we also test the codes at GoogleColab with Python 3.10.12 and default package 
//// Google Drive implementaton hardware: RAM: 83.5GB; GPU: A100 with 40GB
```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.2
torch==2.5.1+cu121
fair-esm==2.0.0
rdkit==2024.03.6
```
### Datasets
The original dataset are also uploaded to [Zenodo](https://zenodo.org/records/13891018).The dataset was downloaded from the our references without any modification. The dataset was divided into training, validation, and test sets as in the original paper. [paper 1](https://www.nature.com/articles/s41467-023-38347-2) and [paper 2](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012100). They can also be downloaded from Google Drive at save at [GoogleDrive link](https://drive.google.com/drive/folders/1op-L1iG55gGDhSCCXT9z62c9kJhoQ769?usp=drive_link).

### Embeddings 
To reduce the computation consumtion of repeated embedding generation for the same molecules and proteins, we separate the embeddings and training procedure.
```
$ python generate_embeddings.py --filename df_train.csv
$ python generate_embeddings.py --filename df_valid.csv
$ python generate_embeddings.py --filename df_test.csv
```
After the embeddings, you will find nine .pt files in your current working directory.

```
df_train_enzy.pt 
df_train_smiles.pt
df_train_label.pt 

df_valid_enzy.pt 
df_valid_smiles.pt 
df_valid_label.pt 

df_test_enzy.pt 
df_test_smiles.pt 
df_test_label.pt
```
A potential debug warning, if your dataset is too large and your RAM is not enough to keep all the generated embeddings, it is suggested to split your dataset into smaller csv files and generate embedding sequentially and concatenate them after embeddings. 
Notice：Extremely long protein sequence will consume too much computation resource and result in CUDA out of memory. We set the maximum length of protein sequence as 5500 with A100 GPU. Feel free to change the maximum seqeunce length to remve long protein sequence if GPU memory is not enough. 

### Training and performance evaluation

```
$ python train.py --train_enzy df_train_enzy.pt --train_smiles df_train_smiles.pt --train_y df_train_label.pt --val_enzy df_valid_enzy.pt --val_smiles df_valid_smiles.pt --val_y df_valid_label.pt --test_enzy df_test_enzy.pt --test_smiles df_test_smiles.pt --test_y df_test_label.pt
```
```
train.py is the file for reproduction;
train_resampling_7_1_2.py is same as the train.py, but redivided the whole dataset into 7:1:2 ratio for train, validation and test.
```

### Reproduction of usage of FusionESP
All the implementation was conducted in Google Colab + Google Drive. The repository provided the original notebook for reproduction (Folder: Notebook for reproduction). 

The original data, embedding files and models are save at [GoogleDrive link](https://drive.google.com/drive/folders/1op-L1iG55gGDhSCCXT9z62c9kJhoQ769?usp=drive_link) (To save computation resource, the embedding operation was stratified based on the protein seqeunce length, Therefore, You will find several files inside the Google Drive folders. Feel free to use them if you want to save the time and resources spent on embeddings).
```
# load the embeddings, an example
ESP_train_df_enzy = torch.load('ESP_train_df_enzy_esm1b_MolFormer.pt')
# in order to load all the embeddings, recommended to refer the jupyter notebook under this reporsitory.
```


```
# Locally reproduction or usage of FusionESP
1. Download the generate_embeddings.py; model.py; train.py
2. Prepare the environment as above (Google Colab has the pre-prepared environment)
3. Run the generate_embeddings.py and train.py subsequently as the command line example above. 
```


### Other data availability
The best model (ESM-2560 + MolFormer training on both experimental evidence-based and phylogenetic evidence-based dataset) listed in Table 3 was also available at [GoogleDrive link](https://drive.google.com/drive/folders/1op-L1iG55gGDhSCCXT9z62c9kJhoQ769?usp=drive_link)



