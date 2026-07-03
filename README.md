# SimBERT-DTI
An extension of the BERT architecture implemented in [ProteinBERT](https://github.com/nadavbra/protein_bert). 
## Description
This extension adds a custom classification head for the prediction of drug-target interactions through the inclusion of a multi-label resolution in the finetuning pipeline. In addition, this architecture was augmented with a similarity-aware regularization loss over both drugs and proteins. 
## Citations
Brandes, N., Ofer, D., Peleg, Y., Rappoport, N. & Linial, M. 
ProteinBERT: A universal deep-learning model of protein sequence and function. 
Bioinformatics (2022). https://doi.org/10.1093/bioinformatics/btac020
