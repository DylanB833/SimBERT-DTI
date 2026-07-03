import os, csv, sys, time
import pandas as pd
import tensorflow as tf
import random
import numpy as np
from IPython.display import display
from tensorflow import keras
from sklearn.model_selection import train_test_split
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len, log
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

BENCHMARKS_DIR = "PATH TO DIRECTORY"

BENCHMARKS = [
    # name, output_type
    ('Drugbank', OutputType(False,'binary',True))
]

settings = {
    'use_similarity': True, # Added setting for passing similarity matrices as additional inputs
    'L1': 0.0, # Tunable hyperparameter for Drug-Similarity term of loss function
    'L2': 0.0, # Tunable hyperparameter for Protein-Similarity term of loss function
    'max_dataset_size': None,
    'max_epochs_per_stage': 40,
    'seq_len': 512,
    'batch_size': 32,
    'final_epoch_seq_len': 1024,
    'initial_lr_with_frozen_pretrained_layers': 1e-02,
    'initial_lr_with_all_layers': 1e-04,
    'final_epoch_lr': 1e-05,
    'dropout_rate': 0.5,
    'training_callbacks': [
        keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
        keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
    ],
}

####### Uncomment for debug mode
# settings['max_dataset_size'] = 500
# settings['max_epochs_per_stage'] = 1

def run_benchmark(benchmark_name, pretraining_model_generator, input_encoder, pretraining_model_manipulation_function = None, fold=0):

    log('========== %s ==========' % benchmark_name)

    output_type = get_benchmark_output_type(benchmark_name)
    log('Output type: %s' % output_type)

    train_set, valid_set, test_set, train_sims, val_sims, label_sims = load_benchmark_dataset(benchmark_name,fold)
    log(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')

    if settings['max_dataset_size'] is not None:
        log('Limiting the training, validation and test sets to %d records each.' % settings['max_dataset_size'])
        train_set = train_set.sample(min(settings['max_dataset_size'], len(train_set)), random_state = 0)
        valid_set = valid_set.sample(min(settings['max_dataset_size'], len(valid_set)), random_state = 0)
        test_set = test_set.sample(min(settings['max_dataset_size'], len(test_set)), random_state = 0)

    train_set['label'] = train_set['label'].astype(str)
    valid_set['label'] = valid_set['label'].astype(str)
    test_set['label'] = test_set['label'].astype(str)

    if output_type.is_categorical:

        if output_type.is_seq:
            unique_labels = sorted(set.union(*train_set['label'].apply(set)) | set.union(*valid_set['label'].apply(set)) | \
                    set.union(*test_set['label'].apply(set)))
        else:
            unique_labels = sorted(set(train_set['label'].unique()) | set(valid_set['label'].unique()) | set(test_set['label'].unique()))

        log('%d unique labels.' % len(unique_labels))
    elif output_type.is_binary:
        unique_labels = [0, 1]
    else:
        unique_labels = None

    if output_type.is_multilabel:
        global_labels = len(train_set['label'].iloc[0])
    else:
        global_labels = 1

    output_spec = OutputSpec(output_type, unique_labels, global_labels, train_sims = train_sims, val_sims = val_sims, label_sims = label_sims, L1=settings['L1'], L2=settings['L2'])
    model_generator = FinetuningModelGenerator(pretraining_model_generator, output_spec, pretraining_model_manipulation_function = \
            pretraining_model_manipulation_function, dropout_rate = settings['dropout_rate'])
    finetune(model_generator, input_encoder, output_spec, train_set['seq'], train_set['label'], valid_set['seq'], valid_set['label'], \
            seq_len = settings['seq_len'], batch_size = settings['batch_size'], max_epochs_per_stage = settings['max_epochs_per_stage'], \
            lr = settings['initial_lr_with_all_layers'], begin_with_frozen_pretrained_layers = True, lr_with_frozen_pretrained_layers = \
            settings['initial_lr_with_frozen_pretrained_layers'], n_final_epochs = 1, final_seq_len = settings['final_epoch_seq_len'], \
            final_lr = settings['final_epoch_lr'], callbacks = settings['training_callbacks'])
    for dataset_name, dataset in [('Test Data', test_set)]:
        log('*** %s performance: ***' % dataset_name)
        allMetrics = {'ROC':None,'PR':None,'F1':None, 'NDCG':[10,50,100],'PREC':[10,50,100],'RECALL':[10,50,100]}

        results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, output_spec, dataset['seq'], dataset['label'], start_seq_len = settings['seq_len'], start_batch_size = settings['batch_size'], metric = allMetrics)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        display(results)

        if confusion_matrix is not None:
            with pd.option_context('display.max_rows', 16, 'display.max_columns', 10):
                log('Confusion matrix:')
                display(confusion_matrix)

    return model_generator, results

def load_benchmark_dataset(raw_benchmark_name, fold):

    benchmark_name = raw_benchmark_name + str(fold)
    train_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.train.csv' % benchmark_name)
    valid_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.valid.csv' % benchmark_name)
    test_set_file_path = valid_set_file_path # Use the validation set as the test set during cross-validation to ensure test set stays blind.

    train_set= pd.read_csv(train_set_file_path)
    test_set = pd.read_csv(test_set_file_path)

    if os.path.exists(valid_set_file_path):
        valid_set = pd.read_csv(valid_set_file_path)
    else:
        log(f'Validation set {valid_set_file_path} missing. Splitting training set instead.')
        train_set, valid_set = train_test_split(train_set, stratify = train_set['label'], test_size = 0.1, random_state = 0)


    if settings['use_similarity']:
            label_path = os.path.join(BENCHMARKS_DIR, "%s_labelsims.csv" % raw_benchmark_name)
            tsims_path = os.path.join(BENCHMARKS_DIR, "%s_sims.train.csv" % benchmark_name)
            vsims_path = os.path.join(BENCHMARKS_DIR, "%s_sims.valid.csv" % benchmark_name)
            label_sims = pd.read_csv(label_path).values
            train_sims = pd.read_csv(tsims_path).values
            val_sims = pd.read_csv(vsims_path).values
    else:
            label_sims = None
            train_sims = None
            val_sims = None

    return train_set, valid_set, test_set, train_sims, val_sims, label_sims

def get_benchmark_output_type(benchmark_name):
    for name, output_type in BENCHMARKS:
        if name == benchmark_name:
            return output_type

pretrained_model_generator, input_encoder = load_pretrained_model()

L1_options = [] # Fill with array of values you wish to test.
L2_options = [] # Fill with array of values you wish to test.

FIELDNAMES = ["AUC","AUPRC","F1","NDCG@10","NDCG@50","NDCG@100","PREC@10","PREC@50","PREC@100","RECALL@10","RECALL@50","RECALL@100"]

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

for fold in range(1,6):

    with open(f"S1_Fold_{fold}.csv","w",newline="") as f:
            writer = csv.writer(f)
            header = ['L1',"L2"] + FIELDNAMES
            writer.writerow(header)

           # Test Hyperparameter Combinations
            for benchmark_name, _ in BENCHMARKS:

                # Compute Baseline for Fold
                settings['L1'] = 0.0
                settings['L2'] = 0.0
                score_array = [0.0] * len(FIELDNAMES)
                model, results = run_benchmark(benchmark_name, pretrained_model_generator, input_encoder, pretraining_model_manipulation_function = \
                      get_model_with_hidden_layers_as_outputs, fold = fold)
                for i,m in enumerate(FIELDNAMES):
                      value = results.get(m,{}).get("All",0)
                      score_array[i] += value

                row = [0.0, 0.0] + score_array
                writer.writerow(row)

                # Compute Hyperparameter Combination Scores
                for L in L1_options:
                      settings['L1'] = L
                      for J in L2_options:
                              settings["L2"] = J
                              score_array = [0.0] * len(FIELDNAMES)
                              model, results = run_benchmark(benchmark_name, pretrained_model_generator, input_encoder, pretraining_model_manipulation_function = \
                                        get_model_with_hidden_layers_as_outputs, fold = fold)
                              for i,m in enumerate(FIELDNAMES):
                                        value = results.get(m,{}).get("All",0)
                                        score_array[i] += value

                              row = [L,J] + score_array
                              writer.writerow(row)

log('Done.')










