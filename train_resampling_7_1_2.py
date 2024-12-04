
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from model import Contrastive_learning_layer

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import argparse

def get_ds(train_data_enzy, train_data_smiles, train_data_y, val_data_enzy, val_data_smiles, val_data_y,test_data_enzy, test_data_smiles, test_data_y, batch_size):
    # Load the saved embeddings_results
    ESP_train_df_enzy = torch.load(train_data_enzy,weights_only=False)
    ESP_val_df_enzy = torch.load(val_data_enzy,weights_only=False)
    ESP_test_df_enzy = torch.load(test_data_enzy,weights_only=False)
    print('Dataset size: protein sequence: ', ESP_train_df_enzy.shape, ESP_val_df_enzy.shape, ESP_test_df_enzy.shape)
    # Load the saved embeddings_results
    ESP_train_df_smiles = torch.load(train_data_smiles,weights_only=False)
    ESP_val_df_smiles = torch.load(val_data_smiles,weights_only=False)
    ESP_test_df_smiles = torch.load(test_data_smiles,weights_only=False)
    print('Dataset size: molecules: ', ESP_train_df_smiles.shape, ESP_val_df_smiles.shape, ESP_test_df_smiles.shape)

    y_train = torch.load(train_data_y,weights_only=False)
    y_val = torch.load(val_data_y,weights_only=False)
    y_test = torch.load(test_data_y,weights_only=False)
    print('Dataset size: label: ', y_train.shape,y_val.shape, y_test.shape)

    train_tensor_dataset = TensorDataset(ESP_train_df_enzy,ESP_train_df_smiles, y_train)
    val_tensor_dataset = TensorDataset(ESP_val_df_enzy,ESP_val_df_smiles, y_val)
    test_tensor_dataset = TensorDataset(ESP_test_df_enzy, ESP_test_df_smiles, y_test)

    # resampling 
    # Combine all datasets into a single dataset
    ESP_enzy = torch.cat([ESP_train_df_enzy, ESP_val_df_enzy, ESP_test_df_enzy], dim=0)
    ESP_smiles = torch.cat([ESP_train_df_smiles, ESP_val_df_smiles, ESP_test_df_smiles], dim=0)
    y = torch.cat([y_train, y_val, y_test], dim=0)

    print('Combined dataset sizes: ')
    print('Protein sequence:', ESP_enzy.shape)
    print('Molecules:', ESP_smiles.shape)
    print('Labels:', y.shape)

    # Shuffle and split the combined dataset into train, validation, and test sets
    num_samples = ESP_enzy.size(0)
    train_size = int(0.7 * num_samples)
    valid_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - valid_size

    indices = torch.randperm(num_samples)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    ESP_train_enzy = ESP_enzy[train_indices]
    ESP_train_smiles = ESP_smiles[train_indices]
    y_train = y[train_indices]

    ESP_val_enzy = ESP_enzy[valid_indices]
    ESP_val_smiles = ESP_smiles[valid_indices]
    y_val = y[valid_indices]

    ESP_test_enzy = ESP_enzy[test_indices]
    ESP_test_smiles = ESP_smiles[test_indices]
    y_test = y[test_indices]

    # Print final dataset sizes
    print('Final dataset sizes:')
    print('Train:', ESP_train_enzy.shape, ESP_train_smiles.shape, y_train.shape)
    print('Validation:', ESP_val_enzy.shape, ESP_val_smiles.shape, y_val.shape)
    print('Test:', ESP_test_enzy.shape, ESP_test_smiles.shape, y_test.shape)

    # Create TensorDatasets
    train_tensor_dataset = TensorDataset(ESP_train_enzy, ESP_train_smiles, y_train)
    val_tensor_dataset = TensorDataset(ESP_val_enzy, ESP_val_smiles, y_val)
    test_tensor_dataset = TensorDataset(ESP_test_enzy, ESP_test_smiles, y_test) 

    # Create TensorDataset and DataLoaders
    # batch_size  # 16
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,  val_loader, test_loader

def run_validation(model, val_loader,loss_fn, device):
    model.eval()
    loss_sum = 0
    num_batch = len(val_loader)
    total_y_true=[]
    total_y_pred=[]
    total_y_prob=[]
    for ESP_val_df_enzy,ESP_val_df_smiles, y_val in val_loader:

        ESP_val_df_enzy = ESP_val_df_enzy.to(device)
        ESP_val_df_smiles = ESP_val_df_smiles.to(device)
        y_val = y_val.squeeze(1).to(device)

        refined_enzy_embed, refined_smiles_embed = model(ESP_val_df_enzy,ESP_val_df_smiles)
        cos_sim = torch.nn.functional.cosine_similarity(refined_enzy_embed, refined_smiles_embed, dim=1)
        loss = loss_fn(cos_sim, y_val).detach().cpu().numpy()
        loss_sum = loss_sum + loss # count all the loss in the training process
        y_pred = (cos_sim > 0.5).float().cpu().numpy() # if score > 0.5, assign label 1 otherwise 0, transfer to cpu as numpy
        total_y_true.append(y_val.cpu().numpy())
        total_y_pred.append(y_pred)
        total_y_prob.append(cos_sim.detach().cpu().numpy())

    loss_sum = loss_sum/num_batch # get the overall average loss (Notice: this method is not 100% accurate)

    arrange_y_true = np.concatenate(total_y_true, axis=0)
    arrange_y_pred = np.concatenate(total_y_pred, axis=0)
    arrange_y_prob = np.concatenate(total_y_prob, axis=0)
    tn,fp,fn,tp = confusion_matrix(arrange_y_true, arrange_y_pred).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    specificity = tn/(tn+fp)
    sensitivity = tp/(tp+fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    bacc = (sensitivity + specificity)/2
    MCC = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    AUC = roc_auc_score(arrange_y_true, arrange_y_prob)
    f1 = 2*precision*recall/(precision+recall)
    print("loss_sum= ",loss_sum, "ACC= ",acc, "bacc= ",bacc, "precision= ",precision,"specificity= ",specificity, "sensitivity= ",sensitivity, "recall= ",recall, "MCC= ",MCC, "AUC= ",AUC, "f1= ",f1)
    return loss_sum, acc, bacc   # , precision, sensitivity, recall, MCC, AUC, f1


def train():
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu" # torch.has_mps or
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate embeddings from a file.")
    parser.add_argument('--train_enzy', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--train_smiles', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--train_y', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--val_enzy', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--val_smiles', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--val_y', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--test_enzy', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--test_smiles', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--test_y', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size during training")
    parser.add_argument('--learning_rate', type=float, default=1e-03, help="batch size during training")
    # Parse the arguments
    args = parser.parse_args()
    
    train_data_enzy = args.train_enzy
    train_data_smiles = args.train_smiles
    train_data_y = args.train_y
    val_data_enzy = args.val_enzy
    val_data_smiles = args.val_smiles
    val_data_y = args.val_y
    test_data_enzy = args.test_enzy
    test_data_smiles = args.test_smiles
    test_data_y = args.test_y
    batch_size = args.batch_size
    lr = args.learning_rate
    
    train_loader,  val_loader, test_loader = get_ds(train_data_enzy, train_data_smiles, train_data_y, val_data_enzy, val_data_smiles, val_data_y,test_data_enzy, test_data_smiles, test_data_y, batch_size)
    # design the model, optimizer and loss function
    model = Contrastive_learning_layer().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr= lr)
    loss_fn = nn.MSELoss().to(device)

    initial_epoch = 0
    best_epoch = -1
    best_accuracy = 0.5
    for epoch in range(initial_epoch, 500):
        torch.cuda.empty_cache()
        model.train()
        with tqdm(train_loader, desc='Processing', unit="batch") as tepoch:
            for ESP_train_df_enzy,ESP_train_df_smiles, y_train in tepoch:
                model.train()
                tepoch.set_description(f"Epoch {epoch}")
                ESP_train_df_enzy = ESP_train_df_enzy.to(device)
                ESP_train_df_smiles = ESP_train_df_smiles.to(device)
                y_train = y_train.squeeze(1).to(device)
                refined_enzy_embed, refined_smiles_embed = model(ESP_train_df_enzy,ESP_train_df_smiles)
                cosine_sim = torch.nn.functional.cosine_similarity(refined_enzy_embed, refined_smiles_embed, dim=1)
                loss = loss_fn(cosine_sim, y_train)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True) #
                tepoch.set_postfix(train_loss=loss)
                # tepoch.set_postfix(val_loss=loss_sum_val, val_accuracy=100. * acc_val, balanced_val_accuracy=100. * bacc_val)
            loss_sum_val,acc_val, bacc_val = run_validation(model,val_loader,loss_fn, device)
            print('Epoch: %d / %d, ############## the best accuracy in val  %.4f at Epoch: %d  ##############'  % (epoch, 500,100 * best_accuracy,best_epoch))
            # print('Performance in Train: Loss: (%.4f); Accuracy (%.2f)' % (loss_sum, 100 * acc))
            print('Performance in Val: Loss: (%.4f); Accuracy (%.2f)' % (loss_sum_val, 100 * acc_val))
            # checkpoint(model, f"epoch-{epoch}.pth")
            if acc_val > best_accuracy: # compare the performance updates at the val set
                best_accuracy = acc_val
                best_epoch = epoch
                torch.save(model, "best_model.pt")         
    # Specify the file path where the entire model is saved
    load_path = 'best_model.pt'
    # Load the entire model
    model_test = torch.load(load_path,weights_only=False)
    # model_test.to('cuda')
    # rename the baseline model
    # torch.save(model_test, 'xxx_best_model.pt')
    print('Model performance in test dataset \n' )
    run_validation(model_test,test_loader,loss_fn, device) # performance evaluation

if __name__ == '__main__':
    train()

