#%%
import pandas as pd
from chembl_webresource_client.new_client import new_client
# %%
targets = new_client.target.search('coronavirus')
targets = pd.DataFrame.from_dict(targets)
query_string = 'target_type=="SINGLE PROTEIN" & organism=="Severe acute respiratory syndrome coronavirus 2"'
target_chem_id = targets.query(query_string)['target_chembl_id'].to_list()[0]

bioactivity = new_client.activity.filter(target_chembl_id = target_chem_id).filter(standard_type="IC50")
bioactivity = pd.DataFrame.from_dict(bioactivity)
bioactivity = bioactivity[(bioactivity['standard_value'].notnull()) & (bioactivity['canonical_smiles'].notnull())]
bioactivity['inhibitor_class'] = bioactivity['standard_value'].apply(lambda x: "strong" if float(x) < 1000.0 else "weak")
bioactivity = bioactivity[['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'inhibitor_class']]
bioactivity.to_csv('../data/raw/cov2_inhibitors.csv', index=False)
# %%

# %%
