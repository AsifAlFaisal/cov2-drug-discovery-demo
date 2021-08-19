#%% Custom Data
## This script is inspired from https://github.com/deepfindr/gnn-project/blob/main/dataset.py

from torch_geometric.data import Data, Dataset
from rdkit import Chem
import numpy as np
import torch
from torch_geometric.transforms import RemoveIsolatedNodes
import pandas as pd
import os
from tqdm import tqdm

class Cov2Data(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.filename = filename
        super(Cov2Data, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.rmv = RemoveIsolatedNodes()
        self.data = pd.read_csv(self.raw_paths[0])
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol['canonical_smiles'])
            node_feat_mat = self._get_node_feature_matrix(mol_obj)
            edge_attr_mat = self._get_edge_attributes(mol_obj)
            adjacency_mat = self._get_edge_index(mol_obj)
            target_class = self._get_classes(mol['inhibitor_class'])
            data = Data(x=node_feat_mat, edge_index=adjacency_mat, 
                        edge_attr=edge_attr_mat, y = target_class, smiles=mol['canonical_smiles'])
            data = self.rmv(data)
            torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))
    
    
    def _get_node_feature_matrix(self, mol):
        # shape [num_nodes, num_node_features]
        all_node_feats = []
        for atom in mol.GetAtoms():
            node_feats = []
            node_feats.append(atom.GetAtomicNum())
            node_feats.append(atom.GetDegree())
            node_feats.append(atom.GetTotalDegree())
            node_feats.append(atom.GetTotalValence())
            node_feats.append(atom.GetTotalNumHs())
            node_feats.append(atom.GetNumRadicalElectrons())
            node_feats.append(atom.GetFormalCharge())
            node_feats.append(atom.GetMass())
            node_feats.append(atom.GetChiralTag())
            node_feats.append(atom.GetHybridization())

            # Node feature matrix
            all_node_feats.append(node_feats)
        
        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_attributes(self, mol):
        all_edge_attr = []
        for bond in mol.GetBonds():
            edge_attr = []
            edge_attr.append(bond.GetBondTypeAsDouble())
            edge_attr.append(bond.GetIsAromatic())
            edge_attr.append(bond.GetIsConjugated())
            edge_attr.append(bond.IsInRing())
            all_edge_attr += [edge_attr, edge_attr]

        all_edge_attr = np.asarray(all_edge_attr)
        return torch.tensor(all_edge_attr, dtype=torch.float)

    def _get_edge_index(self, mol):
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index += [[i,j],[j,i]]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.t().view(2,-1)
        return edge_index

    def _get_classes(self, classes):
        classes = np.asarray([classes])
        return torch.tensor(classes, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

