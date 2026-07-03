# Model
All programs are derived directly from ProteinBERT. Any programs that contain modifications for the DTI extension are documented below. Any program not mentioned contains no significant changes.
## Programs
* finetuning.py
  * Modified to incorporate the multi-label output type and specifications
  * Modified to implement a custom encoding function for multi-label resolution
  * Additional evaluation metrics included, such as NDCG, AUPRC, Precision-Recall, and F1. 
  * Modified evaluation logic to properly handle cold start evaluation
* model_generation.py
  * Modified to incorporate drug/protein similarity matrices, and their relevant holdouts for cold-start
  * Implemented a custom similarity-aware loss function
    * Binary Cross-Entropy loss augmented with protein-neighborhood regularization and drug regularization terms.
  * Custom IndexAwareModel implemented to handle holdouts across batches and epochs accurately. 
  * Modified output layer for multi-label binary classification
