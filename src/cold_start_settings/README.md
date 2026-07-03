# Cold Start Settings
Files necessary for the cold-start pipeline. The split_s*.py files contain the code to process the datasets for input. The tune_setting*.py files contain the code to finetune the L1/L2 hyperparameters. The eval_setting*.py files contain the code to finetune and evaluate the model using specified L1/L2 values.

## Settings
* Setting 1: Protein Cold Start
  * 10% of targets are removed before finetuning.
  * Model is finetuned on 90% of targets and their drug interactions
  * Model is given unseen targets during evaluation and prompted to predict interactions among all drugs.
* Setting 2: Drug Cold Start
  * 10% of drugs are masked before finetuning.
      * _DISCLAIMER:_ Unlike Setting 1, drugs cannot be removed from the finetuning dataset because the model's output dimensionality must remain fixed between finetuning and evaluation. Instead, held-out drugs are masked by replacing all positive labels with zero, effectively treating them as drugs with no known target interactions. This introduces a strong bias towards negative predictions for the masked drugs, making this setting significantly more challenging than a standard cold-start scenario. 
  * Model is finetuned on all targets and their drug interactions
  * Model is given all targets during evaluation and prompted to predict interactions among the masked drugs.
 * Setting 3: Missing Links
   * 10% of interactions are masked before finetuning
   * Model is finetuned on all targets and their drug interactions
   * Model is given all targets during evaluation and prompted to predict interactions for specific masked drugs.
