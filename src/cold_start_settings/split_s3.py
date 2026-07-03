
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import KFold

df = pd.read_csv("Full_Drugbank.csv")
sim = pd.read_csv("Target_sims.csv").to_numpy()


labels = df['label'].values

all_pairs = []

for r, label in enumerate(labels):
	for c, val in enumerate(label):
		all_pairs.append((r,c))

all_pairs = np.array(all_pairs)

np.random.seed(42)
test_size = int(0.1 * len(all_pairs))
test_idx = np.random.choice(len(all_pairs), size=test_size, replace=False)
test_links = all_pairs[test_idx]
remaining = np.ones(len(all_pairs), dtype=bool)
remaining[test_idx] = False
remaining = all_pairs[remaining]

pd.DataFrame(test_links, columns=['row','col']).to_csv('Drugbank_holdouts.test.csv',index=False)

kf = KFold(n_splits=5, shuffle=True, random_state=2)

for fold, (_, holdout) in enumerate(kf.split(remaining)):

	val_links = remaining[holdout]

	train_links = np.ones(len(remaining),dtype=bool)
	train_links[holdout] = False
	train_links = remaining[train_links]

	X = df.copy()
	Y = df.copy()

	X_masked = [list(l) for l in labels]
	Y_masked = [list(l) for l in labels]

	for r, c in np.vstack([val_links, test_links]):
		X_masked[r][c] = '0'

	for r, c in np.vstack([test_links]):
		Y_masked[r][c] = '0'

	X['label'] = [''.join(l) for l in X_masked]
	Y['label'] = [''.join(l) for l in Y_masked]

	X.to_csv(f"Drugbank{fold+1}.train.csv", index=False)
	Y.to_csv(f"Drugbank{fold+1}.valid.csv", index=False)
	pd.DataFrame(sim).to_csv(f"Drugbank_sims{fold+1}.train.csv", index=False)
	pd.DataFrame(sim).to_csv(f"Drugbank_sims{fold+1}.valid.csv", index=False)
	pd.DataFrame(val_links, columns=['row','label']).to_csv(f"Drugbank{fold+1}_holdouts.csv",index=False)
