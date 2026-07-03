import numpy as np
import pandas as pd
import sys

from .shared_utils.util import log
from .tokenization import ADDED_TOKENS_PER_SEQ

class OutputType:

    def __init__(self, is_seq, output_type, is_multilabel = False):
        self.is_seq = is_seq
        self.output_type = output_type
        self.is_multilabel = is_multilabel
        #Sets attributes to True or False based on input
        self.is_numeric = (output_type == 'numeric')
        self.is_binary = (output_type == 'binary')
        self.is_categorical = (output_type == 'categorical')

    def __str__(self):
        #5 output type options: Binary sequence, Categorical sequence, global numeric, global binary, global categorical
        #Numeric sequence not allowed according to encode_seq_Y?
        #Returns the name of the output type
        #Sequence = local, per-residue     Global = Global, whole protein sequence
        if self.is_multilabel:
            return 'multilabel global %s' % self.output_type
        elif self.is_seq:
            return '%s sequence' % self.output_type
        else:
            return 'global %s' % self.output_type

class OutputSpec:

    def __init__(self, output_type, unique_labels = None, global_labels=1, train_sims=None, val_sims=None, label_sims=None, col_holdouts=None, holdouts=None, test_holdouts=None, L1=None, L2=None):
        #Ensures that the given unique labels are correct. If categorical chosen and no unique labels, force user to give new input
        if output_type.is_multilabel:
            if global_labels > 1:
                self.global_labels = global_labels
            else:
                raise ValueError('Illegal number of global labels. Multilabel output requires greater than 1 global label')
        else:
            self.global_labels = global_labels

        if output_type.is_numeric:
            assert unique_labels is None
        elif output_type.is_binary:
            if unique_labels is None:
                unique_labels = [0, 1]
            else:
                assert unique_labels == [0, 1]
        elif output_type.is_categorical:
            assert unique_labels is not None
        else:
            raise ValueError('Unexpected output type: %s' % output_type)
        #Sets output object attributes to user input
        self.output_type = output_type
        self.unique_labels = unique_labels

        #Counts the number of unique labels
        if unique_labels is not None:
            self.n_unique_labels = len(unique_labels)

        # Sets the similarity matrices for use by loss function and its tunable hyperparameter
        if (train_sims is None) != (label_sims is None):
            raise ValueError("A similarity matrix was passed, but two are required. Ensure that both train_sims and label_sims are passed")
        self.train_sims = train_sims
        self.val_sims = val_sims
        self.label_sims = label_sims
        self.L1 = L1
        self.L2 = L2
        self.Cmask = col_holdouts
        self.Hmask = holdouts
        if test_holdouts is None:
                self.THmask = holdouts
        else:
                self.THmask = test_holdouts

def finetune(model_generator, input_encoder, output_spec, train_seqs, train_raw_Y, valid_seqs = None, valid_raw_Y = None, seq_len = 512, batch_size = 32, \
        max_epochs_per_stage = 40, lr = None, begin_with_frozen_pretrained_layers = True, lr_with_frozen_pretrained_layers = None, n_final_epochs = 1, \
        final_seq_len = 1024, final_lr = None, callbacks = []):

    encoded_train_set, encoded_valid_set = encode_train_and_valid_sets(train_seqs, train_raw_Y, valid_seqs, valid_raw_Y, input_encoder, output_spec, seq_len)

    if begin_with_frozen_pretrained_layers:
        log('Training with frozen pretrained layers...')
        model_generator.train(encoded_train_set, encoded_valid_set, seq_len, batch_size, max_epochs_per_stage, lr = lr_with_frozen_pretrained_layers, \
                callbacks = callbacks, freeze_pretrained_layers = True) #Create/train the model using the default(pre-trained) layers

    log('Training the entire fine-tuned model...')
    model_generator.train(encoded_train_set, encoded_valid_set, seq_len, batch_size, max_epochs_per_stage, lr = lr, callbacks = callbacks, \
            freeze_pretrained_layers = False)

    if n_final_epochs > 0:
        log('Training on final epochs of sequence length %d...' % final_seq_len)
        final_batch_size = max(int(batch_size / (final_seq_len / seq_len)), 1)
        encoded_train_set, encoded_valid_set = encode_train_and_valid_sets(train_seqs, train_raw_Y, valid_seqs, valid_raw_Y, input_encoder, output_spec, final_seq_len)
        model_generator.train(encoded_train_set, encoded_valid_set, final_seq_len, final_batch_size, n_final_epochs, lr = final_lr, callbacks = callbacks, \
                freeze_pretrained_layers = False) #Most of the model is trained using seq_len of 512. This final epoch will train using 1024 to capture the handful of sequences that exceed length 512

    model_generator.optimizer_weights = None #Sets weights to None to let evaluate function know that the model is ready for evaluation

def evaluate_by_len(model_generator, input_encoder, output_spec, seqs, raw_Y, start_seq_len = 512, start_batch_size = 32, increase_factor = 2, metric = 'ROC'):

    assert model_generator.optimizer_weights is None

    dataset = pd.DataFrame({'seq': seqs, 'raw_y': raw_Y})

    results = []
    results_names = []
    y_trues = []
    y_preds = []
    output_rows = []

    for len_matching_dataset, seq_len, batch_size in split_dataset_by_len(dataset, start_seq_len = start_seq_len, start_batch_size = start_batch_size, \
            increase_factor = increase_factor):
        #Encode all sequences and annotations in current batch. Do not filter by seq_len
        X, y_true, sample_weights, indices = encode_dataset(len_matching_dataset['seq'], len_matching_dataset['raw_y'], input_encoder, output_spec, \
                seq_len = seq_len, needs_filtering = False)

        if X[0].size == 0:
                continue

        assert set(np.unique(sample_weights)) <= {0.0, 1.0} #Makes sure all sample_weights are 0 or 1
        y_mask = (sample_weights == 1) #Sample weights is a list of all 1's, y_mask is a list of all True values
        model = model_generator.create_model(seq_len) #Calls finetuning create model function without training
        model.save(f'/home/bockdaa/protein_bert/my_models/finetuned_model_{seq_len}',save_format='tf')
        y_pred = model.predict(X, batch_size = batch_size) #Predict labels using built-in keras prediction method
        if output_spec.output_type.is_multilabel:
                y_pred = np.expand_dims(y_pred,axis=-1)

        for i, seq in enumerate(len_matching_dataset['seq']):
                output_rows.append([seq] + y_pred[i].tolist())

        if output_spec.Cmask != None: # Setting 2
                if output_spec.THmask != None: # Setting 2 evaluation
                     y_true = y_true[:, output_spec.THmask]
                     y_pred = y_pred[:, output_spec.THmask]
                     y_true = y_true[y_mask].flatten()
                     y_pred = y_pred[y_mask]
                else:
                     y_true = y_true[:, output_spec.Cmask]
                     y_pred = y_pred[:, output_spec.Cmask]
                     y_true = y_true[y_mask].flatten()
                     y_pred = y_pred[y_mask]

        elif output_spec.THmask != None: # Setting 3

                idx_lookup = {int(idx): i for i, idx in enumerate(len_matching_dataset.index)}
                selected = set(int(r) for (r, _) in output_spec.THmask)
                selected = [(idx_lookup[int(r)],int(r)) for r in selected if int(r) in idx_lookup]
                y_true_list = []
                y_pred_list = []
                for (r, c) in output_spec.THmask:
                        r = int(r)
                        c = int(c)
                        if r in idx_lookup:
                                rb = idx_lookup[r]
                                true_label = int(len_matching_dataset.iloc[rb]['raw_y'][c])
                                y_true_list.append(true_label)
                                y_pred_list.append(y_pred[rb,c,0])

                y_true=np.array(y_true_list)
                y_pred=np.array(y_pred_list)
        else:
                y_true = y_true[y_mask].flatten() #CONTINUE HERE
                y_pred = y_pred[y_mask]

        if output_spec.output_type.is_categorical:
            y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        else:
            y_pred = y_pred.flatten()
        results.append(get_evaluation_results(y_true, y_pred, output_spec, metric))
        results_names.append(seq_len)

        y_trues.append(y_true)
        y_preds.append(y_pred)
    pd.DataFrame(output_rows).to_csv('testset_predictions.csv',index=False)

    y_true = np.concatenate(y_trues, axis = 0)
    y_pred = np.concatenate(y_preds, axis = 0)
    all_results, confusion_matrix = get_evaluation_results(y_true, y_pred, output_spec, metric, return_confusion_matrix = True)
    results.append(all_results)
    results_names.append('All')

    results = pd.DataFrame(results, index = results_names)
    results.index.name = 'Model seq len'

    return results, confusion_matrix

def get_evaluation_results(y_true, y_pred, output_spec, metric, return_confusion_matrix = False):

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, average_precision_score, ndcg_score, f1_score

    results = {}
    results['# records'] = len(y_true)

    if output_spec.output_type.is_numeric:
        results['Spearman\'s rank correlation'] = spearmanr(y_true, y_pred)[0]
        confusion_matrix = None
    else:

        str_unique_labels = list(map(str, output_spec.unique_labels))
        if output_spec.output_type.is_binary:

            y_pred_classes = (y_pred >= 0.5)

            if len(np.unique(y_true)) == 2:
                for m,klist in metric.items():
                        if m == 'ROC':
                                results['AUC'] = roc_auc_score(y_true, y_pred)
                        elif m == 'PR':
                                results['AUPRC'] = average_precision_score(y_true,y_pred)
                        elif m == 'NDCG':
                                for K in klist:
                                        metricName = 'NDCG@' + str(K)
                                        results[metricName] = ndcg_score([y_true], [y_pred], k=K)
                        elif m == 'PREC':
                               for K in klist:
                                        metricName = 'PREC@' + str(K)
                                        relevant = calc_relevant(y_true,y_pred,K)
                                        results[metricName] = relevant / K
                        elif m == 'RECALL':
                               for K in klist:
                                        metricName = 'RECALL@' + str(K)
                                        relevant = calc_relevant(y_true,y_pred,K)
                                        positives = np.count_nonzero(y_true == 1)
                                        results[metricName] = relevant / positives

                        elif m == 'F1':
                               results['F1'] = f1_score(y_true,y_pred_classes)
            else:
                results['AUC'] = np.nan
        elif output_spec.output_type.is_categorical:
            y_pred_classes = y_pred.argmax(axis = -1)
            results['Accuracy'] = accuracy_score(y_true, y_pred_classes)
        else:
            raise ValueError('Unexpected output type: %s' % output_spec.output_type)

        confusion_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred_classes, labels = np.arange(output_spec.n_unique_labels)), index = str_unique_labels, \
                    columns = str_unique_labels)

    if return_confusion_matrix:
        return results, confusion_matrix
    else:
        return results

def calc_relevant(y_true,y_pred,K):
    topK = np.argsort(y_pred)[-K:]
    relevant = 0
    for i in topK:
        if y_true[i] == 1:
                relevant += 1
    return relevant

def encode_train_and_valid_sets(train_seqs, train_raw_Y, valid_seqs, valid_raw_Y, input_encoder, output_spec, seq_len):
    encoded_train_set = encode_dataset(train_seqs, train_raw_Y, input_encoder, output_spec, seq_len = seq_len, needs_filtering = True, \
            dataset_name = 'Training set')

    if valid_seqs is None and valid_raw_Y is None:
        encoded_valid_set = None
    else:
        encoded_valid_set = encode_dataset(valid_seqs, valid_raw_Y, input_encoder, output_spec, seq_len = seq_len, needs_filtering = True, \
                dataset_name = 'Validation set')

    return encoded_train_set, encoded_valid_set

def encode_dataset(seqs, raw_Y, input_encoder, output_spec, seq_len = 512, needs_filtering = True, dataset_name = 'Dataset', verbose = True):
    dataset = pd.DataFrame({'seq': seqs, 'raw_Y': raw_Y, 'sample_idx': np.arange(len(seqs))})
    indices= dataset['sample_idx'] # workaround to ensure the fullset of indices is returned in the case where no filtering is necessary

    if needs_filtering:
        #filters out entries that exceed maximum length
        dataset = filter_dataset_by_len(dataset, seq_len = seq_len, dataset_name = dataset_name, verbose = verbose)
        seqs = dataset['seq']
        raw_Y = dataset['raw_Y']
        indices = dataset['sample_idx']

    X = input_encoder.encode_X(seqs, seq_len) #Takes list where index 0 is a matrix of all encoded sequences, index 1 is a matrix of zeroes of size (# of sequences, 8943)
    Y, sample_weights = encode_Y(raw_Y, output_spec, seq_len = seq_len) #Takes encoded labels Y and an equal size array of sample weights
    return X, Y, sample_weights, indices

def encode_Y(raw_Y, output_spec, seq_len = 512):
    if output_spec.output_type.is_multilabel:
        return encode_multilabel_Y(raw_Y, output_spec)
    elif output_spec.output_type.is_seq: #Calls function for any local(sequence) output type
        return encode_seq_Y(raw_Y, seq_len, output_spec) #Returns encoded Y with sample weights
    elif output_spec.output_type.is_categorical: #Calls function for global categorical output type
        return encode_categorical_Y(raw_Y, output_spec.unique_labels), np.ones(len(raw_Y)) #return an array of encoded labels and a 1D array of ones the same length as the # of labels as sample weights
    elif output_spec.output_type.is_numeric or output_spec.output_type.is_binary: #if global binary or global numeric
        return raw_Y.values.astype(float), np.ones(len(raw_Y)) #return original label values as floating points, and a 1D matrix of 1's the same length as the number of labels as sample weights
    else:
        raise ValueError('Unexpected output type: %s' % output_spec.output_type)

def encode_seq_Y(seqs, seq_len, output_spec):

    numeric = output_spec.output_type.is_numeric

    if numeric:
            Y = np.zeros((len(seqs), seq_len), dtype = float)
    else:
            label_to_index = {str(label): i for i, label in enumerate(output_spec.unique_labels)} #Creates a dictionary where each key is the label, and each value is the location for that label.
            Y = np.zeros((len(seqs), seq_len), dtype = int)                                     #Example: Binary: {0:0,1:1} Categorical {A:0, H:1, C:2,....}

    sample_weights = np.zeros((len(seqs), seq_len))

    for i, seq in enumerate(seqs):
            if numeric:
                    seq = [float(x) for x in seq.split('|')]
            for j, label in enumerate(seq):
                   if not numeric:
                        Y[i, j + 1] = label_to_index[label]
                   else:
                        Y[i, j + 1] = label
            sample_weights[i, 1:(len(seq) + 1)] = 1 #For each row (label) Place a 1 at every index until the label length is met, except for position 0 (start) of a zero matrix.

    if not output_spec.output_type.is_categorical:
            Y = np.expand_dims(Y, axis = -1)        #Transforms the dimensions from 2D to 3D -> Why Binary Only?
            sample_weights = np.expand_dims(sample_weights, axis = -1) #Transforms the dimensions from 2D to 3D

    return Y, sample_weights

def encode_multilabel_Y(labels, output_spec):

    global_labels = output_spec.global_labels

    Y = np.zeros((len(labels), global_labels), dtype = float)

    sample_weights = np.ones(len(labels))

    for i, label_set in enumerate(labels):
        for j, label in enumerate(label_set):
                Y[i,j] = float(label)

    return Y, sample_weights

def encode_categorical_Y(labels, unique_labels):

    label_to_index = {label: i for i, label in enumerate(unique_labels)} #Creates a dictionary where each key is the label, and each value is the location of that label
    Y = np.zeros(len(labels), dtype = int) #zero array of size(# of labels)                               #Example: {A:0, H:1, C:2,....}

    for i, label in enumerate(labels): #loop through each global label
        Y[i] = label_to_index[label] #Find label in dictionary and take index key. Place into zero array
                                     #Example: First label = A -> Y[0] = 0, Second label = C -> Y[1] = 2,...
    return Y

def filter_dataset_by_len(dataset, seq_len = 512, seq_col_name = 'seq', dataset_name = 'Dataset', verbose = True):

    max_allowed_input_seq_len = seq_len - ADDED_TOKENS_PER_SEQ #subtracts 2 tokens from total length (START & END)
    #Creates a dataset of all sequences <= 510 (default)
    filtered_dataset = dataset[dataset[seq_col_name].str.len() <= max_allowed_input_seq_len]
    n_removed_records = len(dataset) - len(filtered_dataset)

    if verbose:
        log('%s: Filtered out %d of %d (%.1f%%) records of lengths exceeding %d.' % (dataset_name, n_removed_records, len(dataset), 100 * n_removed_records / len(dataset), \
                max_allowed_input_seq_len))

    return filtered_dataset

def split_dataset_by_len(dataset, seq_col_name = 'seq', start_seq_len = 512, start_batch_size = 32, increase_factor = 2):

    seq_len = start_seq_len
    batch_size = start_batch_size
    #Default -> all sequences under <= 510, then all sequences <= 1022, <= 2046, ..., until all sequences are evaluated
    while len(dataset) > 0:
        max_allowed_input_seq_len = seq_len - ADDED_TOKENS_PER_SEQ
        len_mask = (dataset[seq_col_name].str.len() <= max_allowed_input_seq_len) #List of True and False values to indicate which indicies are under maximum length
        len_matching_dataset = dataset[len_mask] #Creates list of sequences based on indicies of True values
        yield len_matching_dataset, seq_len, batch_size
        dataset = dataset[~len_mask] #Create list of sequences based on indicies of False values. I.E. removes sequences under current maximum
        seq_len *= increase_factor
        batch_size = max(batch_size // increase_factor, 1)



