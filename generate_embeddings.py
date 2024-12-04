from tqdm import tqdm
import esm
import torch
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
import numpy as np
import pandas as pd
import gc
import argparse

def esm_embeddings_2560(esm2, esm2_alphabet, peptide_sequence_list,device):
  # NOTICE: ESM for embeddings is quite RAM usage, if your sequence is too long,
  #         or you have too many sequences for transformation in a single converting,
  #         you computer might automatically kill the job.
  esm2 = esm2.eval().to(device)
  batch_converter = esm2_alphabet.get_batch_converter()
  # load the peptide sequence list into the bach_converter
  batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
  batch_lens = (batch_tokens != esm2_alphabet.padding_idx).sum(1)
  ## batch tokens are the embedding results of the whole data set
  batch_tokens = batch_tokens.to(device)
  # Extract per-residue representations (on CPU)
  with torch.no_grad():
      # Here we export the last layer of the EMS model output as the representation of the peptides
      # model'esm2_t12_35M_UR50D' only has 12 layers, and therefore repr_layers parameters is equal to 12
      results = esm2(batch_tokens, repr_layers=[36], return_contacts=False)
  token_representations = results["representations"][36].cpu()
  del results, batch_tokens
  torch.cuda.empty_cache()
  gc.collect()
  return token_representations[:,1:-1,:].mean(1)

def MolFormer_embedding(model_smiles, tokenizer, SMILES_list,device):
    inputs = tokenizer(SMILES_list, padding=True, return_tensors="pt").to(device)
    model_smiles = model_smiles.to(device)
    with torch.no_grad():
        outputs = model_smiles(**inputs)
    # NOTICE: if you have several smiles in the list, you will find the average embedding of each token will remain the same
    #           no matter which smiles in side the list, however, the padding will based on the longest smiles,
    #           therefore, the last hidden state representation shape:[len, 768] will change for the same smiles in difference smiles list.
    return outputs.pooler_output.cpu() # shape is [len_list, 768] ; torch tensor;

def main():
    # """CPU or GPU."""
    device = "cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu" # torch.has_mps or
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with Nvidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate embeddings from a file.")
    parser.add_argument('--filename', type=str, required=True, help="Path to the input file.")
    # Parse the arguments
    args = parser.parse_args()
    ESP_train_df = pd.read_csv(args.filename, header= 0)
    model_smiles = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    
    # select the ESM model for embeddings (you can select you desired model from https://github.com/facebookresearch/esm)
    # NOTICE: if you choose other model, the following model architecture might not be very compitable
    #         bseides,please revise the correspdoning parameters in esm_embeddings function (layers for feature extraction)
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    # generate the peptide embeddings
    embeddings_results_enzy = []
    embeddings_results_smiles = []
    embeddings_results_label = []
    for i in tqdm(range(ESP_train_df.shape[0]), desc="Processing Sequences"):
        # the setting is just following the input format setting in ESM model, [name,sequence]
        seq_enzy = ESP_train_df['Protein sequence'].iloc[i]
        seq_smiles = ESP_train_df['SMILES'].iloc[i]
        if len(seq_enzy) < 5500:
            tuple_sequence = tuple(['protein',seq_enzy])
            peptide_sequence_list = []
            peptide_sequence_list.append(tuple_sequence) # build a summarize list variable including all the sequence information
            # employ ESM model for converting and save the converted data in csv format
            one_seq_embeddings = esm_embeddings_2560(model, alphabet, peptide_sequence_list,device)
            embeddings_results_enzy.append(one_seq_embeddings)
            # the smiles embeddings
            smiles_list = []
            smiles_list.append(Chem.CanonSmiles(seq_smiles)) # build a summarize list variable including all the sequence information
            # employ ESM model for converting and save the converted data in csv format
            one_seq_embeddings = MolFormer_embedding(model_smiles, tokenizer, smiles_list,device)
            embeddings_results_smiles.append(one_seq_embeddings)

            # record the lable info
            label = torch.tensor(ESP_train_df['output'].iloc[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            embeddings_results_label.append(label)

    output_file_enzy = args.filename.split('.')[0] + '_enzy.pt'
    output_file_smiles = args.filename.split('.')[0] + '_smiles.pt'
    output_file_label = args.filename.split('.')[0] + '_label.pt'
    embeddings_results_enzy_torch = torch.cat(embeddings_results_enzy, dim=0)
    torch.save(embeddings_results_enzy_torch, output_file_enzy)

    embeddings_results_smiles_torch = torch.cat(embeddings_results_smiles, dim=0)
    torch.save(embeddings_results_smiles_torch, output_file_smiles)

    embeddings_results_label_torch = torch.cat(embeddings_results_label, dim=0)
    torch.save(embeddings_results_label_torch, output_file_label)
    
if __name__ == '__main__':
    main()
