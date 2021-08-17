#%% Imports
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np
import torch
# %% Creating Custom PyTorch Geometric Data Object
class Cov2Data:
    def __init__(self):
        pass

    def getData(self,smiles, target):
        self.smiles = smiles
        self.target = target
        mol_obj = Chem.MolFromSmiles(self.smiles)
        node_feat_mat = self._get_node_feature_matrix(mol_obj)
        edge_attr_mat = self._get_edge_attributes(mol_obj)
        adjacency_mat = self._get_edge_index(mol_obj)
        target_class = self._get_classes(self.target)
        data = Data(X=node_feat_mat, edge_index=adjacency_mat, edge_attr=edge_attr_mat, y = target_class, smiles=self.smiles)
        return data
    
    
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
