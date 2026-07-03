
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import KFold

df = pd.read_csv("Full_Drugbank.csv")
sim = pd.read_csv("Target_sims.csv").to_numpy()

test_set = df.sample(frac=0.1,random_state=42)
test_set.to_csv("Drugbank.test.csv",index=False)
test_idx = test_set.index

train_df = df.drop(index=test_idx)

mask = np.ones(sim.shape[0], dtype=bool)
mask[test_idx] = False
train_sims = sim[np.ix_(mask,mask)]

kf = KFold(n_splits=5, shuffle=True, random_state=3)

for fold, (train_idx, test_idx) in enumerate(kf.split(train_df)):

	X = train_df.iloc[train_idx].reset_index(drop=True)
	Y = train_df.iloc[test_idx].reset_index(drop=True)

	Sx = train_sims[np.ix_(train_idx, train_idx)]
	Sy = train_sims[np.ix_(test_idx, test_idx)]

	X.to_csv(f"Drugbank{fold+1}.train.csv",index=False)
	Y.to_csv(f"Drugbank{fold+1}.valid.csv",index=False)

	pd.DataFrame(Sx).to_csv(f"Drugbank{fold+1}_sims.train.csv",index=False)
	pd.DataFrame(Sy).to_csv(f"Drugbank{fold+1}_sims.valid.csv",index=False)
