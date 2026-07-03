from numbers import Number
import pickle
import tensorflow as tf
import numpy as np
import sys
import tensorflow.keras.backend as K

from tensorflow import keras

from .shared_utils.util import log
from .tokenization import additional_token_to_index, n_tokens, tokenize_seq
    
class ModelGenerator:

    def __init__(self, optimizer_class = keras.optimizers.Adam, lr = 2e-04, other_optimizer_kwargs = {}, model_weights = None, optimizer_weights = None):
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.other_optimizer_kwargs = other_optimizer_kwargs
        self.model_weights = model_weights
        self.optimizer_weights = optimizer_weights
        
    def train(self, encoded_train_set, encoded_valid_set, seq_len, batch_size, n_epochs, lr = None, callbacks = [], **create_model_kwargs):
        train_X, train_Y, train_sample_weigths, indices = encoded_train_set
        self.dummy_epoch = (_slice_arrays(train_X, slice(0, 1)), _slice_arrays(train_Y, slice(0, 1)))
        model = self.create_model(seq_len, **create_model_kwargs)
        if lr is not None:
            model.optimizer.lr = lr

        X_dataset = tf.data.Dataset.from_tensor_slices(((train_X[0],train_X[1]), train_Y, train_sample_weigths, indices)).batch(batch_size)
        val_X, val_Y, val_sw, val_indices = encoded_valid_set
        V_dataset = tf.data.Dataset.from_tensor_slices(((val_X[0],val_X[1]), val_Y, val_sw, val_indices)).batch(batch_size)
        model.fit(X_dataset, validation_data=V_dataset, epochs=n_epochs, callbacks=callbacks)
        self.update_state(model)

    def update_state(self, model):
        self.model_weights = copy_weights([w.numpy() for w in model.variables])
        self.optimizer_weights = copy_weights([w.numpy() for w in model.optimizer.variables()])
        
    def _init_weights(self, model):
    
   #     if self.optimizer_weights is not None:
            # For some reason keras requires this strange little hack in order to properly initialize a new model's optimizer, so that
            # the optimizer's weights can be reloaded from an existing state.
    #        self._train_for_a_dummy_epoch(model)
            
        if self.model_weights is not None:
            model.set_weights(copy_weights(self.model_weights))
        
        if self.optimizer_weights is not None:
            if len(self.optimizer_weights) == len(model.optimizer.variables()):
                model.optimizer.set_weights(copy_weights(self.optimizer_weights))
            else:
                log('Incompatible number of optimizer weights - will not initialize them.')
     # Dummy epoch temporarily removed when using IndexAwareModel. Add logic to handle this automatically later.
#    def _train_for_a_dummy_epoch(self, model):
 #       X, Y = self.dummy_epoch
  #      model.fit(X, Y, batch_size = 1, verbose = 0)
        
class PretrainingModelGenerator(ModelGenerator):

    def __init__(self, create_model_function, n_annotations, create_model_kwargs = {}, optimizer_class = keras.optimizers.Adam, lr = 2e-04, other_optimizer_kwargs = {}, \
            annots_loss_weight = 1, model_weights = None, optimizer_weights = None):
        
        ModelGenerator.__init__(self, optimizer_class = optimizer_class, lr = lr, other_optimizer_kwargs = other_optimizer_kwargs, model_weights = model_weights, \
                optimizer_weights = optimizer_weights)
        
        self.create_model_function = create_model_function
        self.n_annotations = n_annotations
        self.create_model_kwargs = create_model_kwargs
        self.annots_loss_weight = annots_loss_weight
        
    def create_model(self, seq_len, compile = True, init_weights = True):
        clear_session()
        model = self.create_model_function(seq_len, n_tokens, self.n_annotations, **self.create_model_kwargs)
        
        if compile:
            model.compile(optimizer =self.optimizer_class(learning_rate = self.lr, **self.other_optimizer_kwargs), loss = ['sparse_categorical_crossentropy', 'binary_crossentropy'], \
                    loss_weights = [1, self.annots_loss_weight])
        
        if init_weights:
            self._init_weights(model)
        
        return model
        

class FinetuningModelGenerator(ModelGenerator):

    def __init__(self, pretraining_model_generator, output_spec, pretraining_model_manipulation_function = None, dropout_rate = 0.5, optimizer_class = None, \
            lr = None, other_optimizer_kwargs = None, model_weights = None, optimizer_weights = None):
        
        if other_optimizer_kwargs is None:
            if optimizer_class is None:
                other_optimizer_kwargs = pretraining_model_generator.other_optimizer_kwargs
            else:
                other_optimizer_kwargs = {}
        
        if optimizer_class is None:
            optimizer_class = pretraining_model_generator.optimizer_class
            
        if lr is None:
            lr = pretraining_model_generator.lr
            
        ModelGenerator.__init__(self, optimizer_class = optimizer_class, lr = lr, other_optimizer_kwargs = other_optimizer_kwargs, model_weights = model_weights, \
                optimizer_weights = optimizer_weights)
        
        self.pretraining_model_generator = pretraining_model_generator
        self.output_spec = output_spec
        self.pretraining_model_manipulation_function = pretraining_model_manipulation_function
        self.dropout_rate = dropout_rate
        self.L1 = self.output_spec.L1 # Added hyperparamter for tuning drug term similarity
        self.L2 = self.output_spec.L2 # Added hyperparamter for tuning protein term similarity
        self.D = tf.constant(self.output_spec.label_sims, dtype=tf.float32) # Drug similarity matrix
        self.Pt = tf.constant(self.output_spec.train_sims, dtype=tf.float32) # Protein similarity matrix for training
        self.Pv = tf.constant(self.output_spec.val_sims, dtype=tf.float32) # Protein similarity matrix for validation
        self.Cmask = self.output_spec.Cmask
        self.Hmask = self.output_spec.Hmask

    def compute_sim_loss(self, y_true, y_pred_logits, indices, use_val=False, batch_hmask=None):
        # Custom loss function designed to take label and protein similarity matrices as additional inputs.
        # Binary Cross Entropy with Sigmoid Activation
                y_true = tf.cast(y_true, tf.float32)

                if use_val and self.Cmask != None: # If using column holdouts, then extract the y_true and y_pred that are relevant. Before BCE to calculate using smaller label space.
                        y_true = tf.gather(y_true, self.Cmask, axis=1)
                        y_pred_logits = tf.gather(y_pred_logits, self.Cmask, axis=1)

                bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred_logits)

                if use_val and batch_hmask != None: #If using random holdouts, then calculate AFTER BCE instead of before. We still need the full label space. Simply hide these pairs from the loss function instead.
                        hmask = tf.cast(batch_hmask, tf.float32)
                        bce = bce * hmask
                        safe_denom = tf.maximum(tf.reduce_sum(hmask), 1e-8)
                        bce_term = tf.reduce_sum(bce) / safe_denom
                else:
                        bce_term = tf.reduce_mean(bce)

        # Drug Term
                # Formula
                if use_val and self.Cmask != None:
                        D = tf.gather(tf.gather(self.D, self.Cmask, axis=0), self.Cmask, axis=1)
                else:
                        D = self.D

                d_diff = y_pred_logits[:, :, None] - y_pred_logits[:, None, :] # Calculate pairwise differences per row
                sq_ddiff = tf.square(d_diff) # Square each pairwise difference
                weighted_ddiff = sq_ddiff * D # Multiply each score by drug similarity
                drug_term = tf.reduce_mean(weighted_ddiff)

        # Protein Term

                p_diff = y_pred_logits[:, None, :] - y_pred_logits[None, :, :] # Calculate pairwise differences per column
                sq_pdiff = tf.square(p_diff) # Square each pairwise difference

                if use_val:
                        P_batch = tf.gather(tf.gather(self.Pv, indices, axis=0),indices,axis=1)[:,:,None] # Extract the submatrix for validation
                else:
                        P_batch = tf.gather(tf.gather(self.Pt, indices, axis=0),indices,axis=1)[:,:,None] # Extract the submatrix for training batch

                weighted_pdif = sq_pdiff * P_batch # Multiply each score by protein similarity
                protein_term = tf.reduce_mean(weighted_pdif)

                return bce_term + (self.L1*drug_term) + (self.L2*protein_term)

    def create_model(self, seq_len, freeze_pretrained_layers = False):
        
        model = self.pretraining_model_generator.create_model(seq_len, compile = False, init_weights = (self.model_weights is None))
            
        if self.pretraining_model_manipulation_function is not None:
            model = self.pretraining_model_manipulation_function(model)
        if freeze_pretrained_layers:
            for layer in model.layers:
                layer.trainable = False
        
        model_inputs = model.input

        pretraining_output_seq_layer, pretraining_output_annoatations_layer = model.output
        last_hidden_layer = pretraining_output_seq_layer if self.output_spec.output_type.is_seq else pretraining_output_annoatations_layer
        last_hidden_layer = keras.layers.Dropout(self.dropout_rate)(last_hidden_layer)

        if not (self.output_spec.train_sims is None):
            output_layer = keras.layers.Dense(self.output_spec.global_labels, activation = None)(last_hidden_layer)
            model = IndexAwareModel(loss_fn=self.compute_sim_loss, Hmask=self.Hmask, inputs=model_inputs, outputs=output_layer)
            model.compile(optimizer =self.optimizer_class(learning_rate = self.lr, **self.other_optimizer_kwargs))
            self._init_weights(model)
            return model

        if self.output_spec.output_type.is_categorical:
            output_layer = keras.layers.Dense(len(self.output_spec.unique_labels), activation = 'softmax')(last_hidden_layer)
            loss = 'sparse_categorical_crossentropy'
        elif self.output_spec.output_type.is_binary:
                output_layer = keras.layers.Dense(self.output_spec.global_labels, activation = 'sigmoid')(last_hidden_layer)
                loss = 'binary_crossentropy'

        elif self.output_spec.output_type.is_numeric:
            output_layer = keras.layers.Dense(self.output_spec.global_labels, activation = None)(last_hidden_layer)
            loss = 'mse'
        else:
            raise ValueError('Unexpected global output type: %s' % self.output_spec.output_type)
                
        model = keras.models.Model(inputs = model_inputs, outputs = output_layer)
        model.compile(loss=loss, optimizer =self.optimizer_class(learning_rate = self.lr, **self.other_optimizer_kwargs))
        
        self._init_weights(model)
                
        return model
                        
class IndexAwareModel(tf.keras.Model):
    def __init__(self, loss_fn, Hmask, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        if Hmask != None:
                self.Hmask_rows = tf.constant([r for r, c in Hmask], dtype=tf.int32)
                self.Hmask_cols = tf.constant([c for r, c in Hmask], dtype=tf.int32)
                self.use_hmask = True
        else:
                self.use_hmask = False

    def train_step(self, data):
        x, y, sample_weight, indices = data

        with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.loss_fn(y, y_pred, indices)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, y, sample_weight, indices = data
        y_pred = self(x,training=False)
        y = tf.cast(y, tf.float32)
        indices = tf.cast(indices, tf.int32)

        if self.use_hmask:
                batch_size = tf.shape(indices)[0]
                matches = tf.equal(self.Hmask_rows[:, None], indices[None, :])
                where = tf.where(matches)
                hold_ids = where[:, 0]
                batch_ids = tf.cast(where[:, 1], tf.int32)
                label_ids = tf.gather(self.Hmask_cols, hold_ids)
                scatter_idx = tf.stack([batch_ids, label_ids], axis=1)
                batch_hmask = tf.zeros((batch_size, tf.shape(y_pred)[1]), dtype=tf.float32)
                batch_hmask = tf.cond(tf.shape(scatter_idx)[0] > 0, lambda: tf.tensor_scatter_nd_update(batch_hmask, scatter_idx, tf.ones(tf.shape(scatter_idx)[0], tf.float32)),lambda: batch_hmask)

                loss = self.loss_fn(y, y_pred, indices, use_val=True, batch_hmask=batch_hmask)
        else:
                loss = self.loss_fn(y, y_pred, indices, use_val=True)

        return {"loss": loss}

class InputEncoder:

    def __init__(self, n_annotations):
        self.n_annotations = n_annotations

    def encode_X(self, seqs, seq_len):
        return [tokenize_seqs(seqs, seq_len), np.zeros((len(seqs), self.n_annotations), dtype = np.int8)]
        
def load_pretrained_model_from_dump(dump_file_path, create_model_function, create_model_kwargs = {}, optimizer_class = keras.optimizers.Adam, lr = 2e-04, \
        other_optimizer_kwargs = {}, annots_loss_weight = 1, load_optimizer_weights = False):
    
    with open(dump_file_path, 'rb') as f:
        n_annotations, model_weights, optimizer_weights = pickle.load(f)
        
    if not load_optimizer_weights:
        optimizer_weights = None
    
    model_generator = PretrainingModelGenerator(create_model_function, n_annotations, create_model_kwargs = create_model_kwargs, optimizer_class = optimizer_class, lr = lr, \
            other_optimizer_kwargs = other_optimizer_kwargs, annots_loss_weight = annots_loss_weight, model_weights = model_weights, optimizer_weights = optimizer_weights)
    input_encoder = InputEncoder(n_annotations)
    
    return model_generator, input_encoder

def tokenize_seqs(seqs, seq_len):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in map(tokenize_seq, seqs)], dtype = np.int32)
    
def clear_session():
    import tensorflow.keras.backend as K
    K.clear_session()
    
def copy_weights(weights):
    return [_copy_number_or_array(w) for w in weights]
    
def _copy_number_or_array(variable):
    if isinstance(variable, np.ndarray):
        return variable.copy()
    elif isinstance(variable, Number):
        return variable
    else:
        raise TypeError('Unexpected type %s' % type(variable))
    
def _slice_arrays(arrays, slicing):
    if isinstance(arrays, list) or isinstance(arrays, tuple):
        return [array[slicing] for array in arrays]
    else:
        return arrays[slicing]
