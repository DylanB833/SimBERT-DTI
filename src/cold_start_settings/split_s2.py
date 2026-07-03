
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import KFold

df = pd.read_csv("Full_Drugbank.csv")
sim = pd.read_csv("Target_sims.csv").to_numpy()

labels = df['label'].values
L = len(labels[0])
all_cols = np.arange(L)

np.random.seed(42)
test_cols = np.random.choice(all_cols, size=int(0.1 * L), replace=False)

pd.DataFrame({'test_cols': test_cols}).to_csv("Drugbank_label_holdouts.test.csv",index=False)

remaining = np.setdiff1d(all_cols, test_cols)

kf = KFold(n_splits=5, shuffle=True, random_state=3)

for fold, (_, holdouts) in enumerate(kf.split(remaining)):

	val_cols = remaining[holdouts]
	Y = df.copy()
	X = df.copy()

	def mask_train(label):
		s = list(label)
		for i in np.concatenate([val_cols, test_cols]):
			s[i] = '0'
		return ''.join(s)

	X_train = df.copy()
	X_train['label'] = [mask_train(l) for l in labels]

	def mask_val(label):
		s = list(label)
		for i in test_cols:
			s[i] = '0'
		return ''.join(s)

	X['label'] = [mask_train(j) for j in labels]
	Y['label'] = [mask_val(j) for j in labels]

	X.to_csv(f"Drugbank.train.csv",index=False)
	Y.to_csv(f"Drugbank.valid.csv",index=False)

	pd.DataFrame(sim).to_csv(f"Drugbank_sims.train.csv",index=False)
	pd.DataFrame(sim).to_csv(f"Drugbank_sims.valid.csv",index=False)

	pd.DataFrame({'holdouts': val_cols}).to_csv(f"Drugbank_labelmask.csv",index=False)
	break

