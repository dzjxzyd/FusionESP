import torch
import torch.nn as nn

class Contrastive_learning_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enzy_refine_layer_1 = nn.Linear(2560, 2560) # W1 and b
        self.smiles_refine_layer_1 = nn.Linear(768, 768) # W1 and b
        self.enzy_refine_layer_2 = nn.Linear(2560, 128) # W1 and b
        self.smiles_refine_layer_2 = nn.Linear(768, 128) # W1 and b

        self.relu = nn.ReLU()
        self.batch_norm_enzy = nn.BatchNorm1d(2560)
        self.batch_norm_smiles = nn.BatchNorm1d(768)
        self.batch_norm_shared = nn.BatchNorm1d(128)

    def forward(self, enzy_embed, smiles_embed):
        refined_enzy_embed = self.enzy_refine_layer_1(enzy_embed)
        refined_smiles_embed = self.smiles_refine_layer_1(smiles_embed)

        refined_enzy_embed = self.batch_norm_enzy(refined_enzy_embed)
        refined_smiles_embed = self.batch_norm_smiles(refined_smiles_embed)

        refined_enzy_embed = self.relu(refined_enzy_embed)
        refined_smiles_embed = self.relu(refined_smiles_embed)

        refined_enzy_embed = self.enzy_refine_layer_2(refined_enzy_embed)
        refined_smiles_embed = self.smiles_refine_layer_2(refined_smiles_embed)

        refined_enzy_embed = self.batch_norm_shared(refined_enzy_embed)
        refined_smiles_embed = self.batch_norm_shared(refined_smiles_embed)
        refined_enzy_embed = torch.nn.functional.normalize(refined_enzy_embed, dim=1)
        refined_smiles_embed = torch.nn.functional.normalize(refined_smiles_embed, dim=1)

        return refined_enzy_embed, refined_smiles_embed
