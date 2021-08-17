#%%
import pandas as pd
from custom_data import Cov2Data
from torch_geometric.data import DataLoader
# %%
df = pd.read_csv('../data/raw/cov2_inhibitors.csv')
# %%
DataObj = Cov2Data()
DataList = [DataObj.getData(mol, target) for mol, target in zip(df['canonical_smiles'], df['inhibitor_class'])]
loader = DataLoader(DataList, batch_size=4)
# %%
