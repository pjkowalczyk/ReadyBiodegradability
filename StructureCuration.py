import pandas as pd
import numpy as np
from rdkit import Chem

df = pd.read_csv('C:/Users/us16120/Projects/Cheminformatics/readybiodegradability/data/processed/alles.csv', header = 0)
df = df.drop(['Unnamed: 0'], axis = 1)

for i, row in df.iterrows():
    df.loc[i, 'CanonicalSMILES'] = Chem.MolToSmiles(Chem.MolFromSmiles(row['SMILES']))
    
for i, row in df.iterrows():
    df.loc[i, 'InChI'] = Chem.MolToInchiKey(Chem.MolFromSmiles(row['CanonicalSMILES']))
     
df.sample(5).head()
