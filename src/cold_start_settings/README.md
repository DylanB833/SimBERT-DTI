# Cold Start Settings
Files necessary for the cold-start pipeline. The split_s*.py files contain the code to process the datasets for input. The tune_setting*.py files contain the code to finetune the L1/L2 hyperparameters. The eval_setting*.py files contain the code to finetune and evaluate the model using specified L1/L2 values.

## Settings
* Setting 1: Protein Cold Start
  * 10% of targets are removed before finetuning.
  * Model is finetuned on targets and all of their drug interactions.
  * Model is given unseen targets during evaluation and prompted to predict interactions among 1,482 drugs.
* Setting 2: Drug Cold Start
  * 10% of drugs are masked before finetuning.
      * _DISCLAIMER:_ Unlike Setting 1, drugs cannot be removed from the finetuning dataset because the model's output dimensionality must remain fixed between finetuning and evaluation. Insetad, held-out drugs are masked by replacing all positive labels with zero, effectively treating them as drugs with no known target interactions. 
  * Model is finetuned on all targets and all of their drug interactions
