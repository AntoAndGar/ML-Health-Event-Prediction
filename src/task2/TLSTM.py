import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# from sklearn.model_selection import train_test_split

SEED = 0


def create_dataset(
    df_anagrafica,
    df_diagnosi,
    df_esami_par,
    df_esami_par_cal,
    df_esami_stru,
    df_pre_diab_farm,
    df_pre_diab_no_farm,
    df_pre_no_diab,
):
    # Adding 'tipo' column directly to each DataFrame
    df_esami_par["tipo"] = "esame parametri"
    df_esami_par_cal["tipo"] = "esame parametri calcolati"
    df_esami_stru["tipo"] = "esame strumentale"
    df_pre_diab_farm["tipo"] = "prescrizione farmaco diabate"
    df_pre_diab_no_farm["tipo"] = "prescrizione non-farmaco diabete"
    df_pre_no_diab["tipo"] = "prescrizione non-diabete"
    df_diagnosi["tipo"] = "diagnosi"

    # Concatenate dataframes
    dfs = [
        df_esami_par,
        df_esami_par_cal,
        df_esami_stru,
        df_pre_diab_farm,
        df_pre_diab_no_farm,
        df_pre_no_diab,
        df_diagnosi,
    ]
    final_df = pd.concat(dfs, ignore_index=True)

    # Drop unnecessary columns because the df is too big
    df_anagrafica = df_anagrafica.drop(
        columns=[
            "annonascita",
            "annoprimoaccesso",
            "annodecesso",
            "annodiagnosidiabete",
        ]
    )
    # Merge with df_anagrafica
    final_df = final_df.merge(
        df_anagrafica, on=["idana", "idcentro"], how="inner"
    ).sort_values(by=["data"])

    grouped = final_df.groupby(["idana", "idcentro"])
    final_df["delta_events"] = grouped["data"].diff().dt.days.fillna(0)
    final_df["delta_events"] = 1 / np.log(np.e + final_df["delta_events"])

    # Mapping for 'sesso' column (use astype to save memory)
    final_df["sesso"] = (
        final_df["sesso"].map({"M": 0, "F": 1}, na_action="ignore").astype("Int8")
    )

    # Mapping for 'valore' column (use astype to save memory)
    valore_mapping = {"N": 0, "P": 1, "S": 2}
    final_df["valore"] = (
        final_df["valore"].map(valore_mapping, na_action="ignore").astype("Int8")
    )

    # Mapping for other columns
    columns_to_map = [
        "codiceamd",
        "codiceatc",
        "codicestitch",
        "descrizionefarmaco",
        "tipo",
    ]
    for col in columns_to_map:
        mapping = {k: v for v, k in enumerate(final_df[col].unique())}
        final_df[col] = final_df[col].map(mapping, na_action="ignore").astype("Int16")

    # Convert 'valore' to numeric
    final_df["valore"] = pd.to_numeric(final_df["valore"], errors="coerce")

    # Fill NaNs with -1
    final_df = final_df.fillna(-1)

    # Check for non-numeric 'valore' values
    non_numeric_valores = final_df["valore"].unique()
    if any(isinstance(i, str) for i in non_numeric_valores):
        raise ValueError("Value not converted to numeric.")

    return final_df


def get_dataset_partitions_tf(
    ds, ds_size, train_split=0.8, val_split=0.2, shuffle=True, shuffle_size=10000
):
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=SEED)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)

    return train_ds, val_ds


def create_tensor_dataset(input_df):
    # Calculate the input length directly
    len_input = len(input_df.columns) - 4

    # Group by "idana" and "idcentro" and sort each group by "data"
    grouped = input_df.groupby(["idana", "idcentro"], group_keys=True)

    inputs = []
    labels = []
    elapsed_time = []
    max_history_len = 0

    for _, group in grouped:
        patient_history = group.sort_values(by=["data"])
        labels.append(
            patient_history["label"].iloc[0]
        )  # Use iloc to access the first item
        elapsed_time.append(patient_history["delta_events"].to_numpy(dtype=np.float32))
        patient_history = patient_history.drop(
            columns=["idana", "idcentro", "label", "data", "delta_events"]
        )
        nested_list = patient_history.to_numpy(dtype=np.float32)
        inputs.append(nested_list)

        max_history_len = max(
            max_history_len, nested_list.shape[0]
        )  # Update max_history_len

    # Pad sequences using pad_sequences
    padded_inputs = pad_sequences(
        inputs, maxlen=512, dtype=np.float32, padding="pre", value=-100.0
    )
    padded_elapsed_time = pad_sequences(
        elapsed_time, maxlen=512, dtype=np.float32, padding="pre", value=0.0
    )
    print(padded_inputs.shape)
    print(padded_elapsed_time.shape)

    bool_array = np.array(labels, dtype=np.float32)
    print(bool_array)
    print(bool_array.shape)
    bool_array = np.repeat(np.expand_dims(bool_array, axis=1), 512, axis=1)
    print(bool_array)
    print(bool_array.shape)

    # Printing unique values in bool_array
    unique_values, value_counts = np.unique(bool_array, return_counts=True)
    print("Unique values in bool_array:")
    print(unique_values)
    print("Value counts:")
    print(value_counts)
    print("Elapsed time: ", len(elapsed_time))

    return padded_inputs, bool_array, padded_elapsed_time


# from https://github.com/illidanlab/T-LSTM/blob/master/TLSTM.py
class TLSTM(object):
    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        return tf.compat.v1.get_variable(
            name,
            shape=[input_dim, output_dim],
            initializer=tf.random_normal_initializer(0.0, std),
            regularizer=reg,
        )

    def init_bias(self, output_dim, name):
        return tf.compat.v1.get_variable(
            name, shape=[output_dim], initializer=tf.constant_initializer(1.0)
        )

    def no_init_weights(self, input_dim, output_dim, name):
        return tf.compat.v1.get_variable(name, shape=[input_dim, output_dim])

    def no_init_bias(self, output_dim, name):
        return tf.compat.v1.get_variable(name, shape=[output_dim])

    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, train):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # [batch size x seq length x input dim]
        tf.compat.v1.disable_eager_execution()
        self.input = tf.compat.v1.placeholder(
            "float", shape=[None, None, self.input_dim]
        )
        self.labels = tf.compat.v1.placeholder("float", shape=[None, output_dim])
        self.time = tf.compat.v1.placeholder("float", shape=[None, None])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)

        if train == 1:
            self.Wi = self.init_weights(
                self.input_dim, self.hidden_dim, name="Input_Hidden_weight", reg=None
            )
            self.Ui = self.init_weights(
                self.hidden_dim, self.hidden_dim, name="Input_State_weight", reg=None
            )
            self.bi = self.init_bias(self.hidden_dim, name="Input_Hidden_bias")

            self.Wf = self.init_weights(
                self.input_dim, self.hidden_dim, name="Forget_Hidden_weight", reg=None
            )
            self.Uf = self.init_weights(
                self.hidden_dim, self.hidden_dim, name="Forget_State_weight", reg=None
            )
            self.bf = self.init_bias(self.hidden_dim, name="Forget_Hidden_bias")

            self.Wog = self.init_weights(
                self.input_dim, self.hidden_dim, name="Output_Hidden_weight", reg=None
            )
            self.Uog = self.init_weights(
                self.hidden_dim, self.hidden_dim, name="Output_State_weight", reg=None
            )
            self.bog = self.init_bias(self.hidden_dim, name="Output_Hidden_bias")

            self.Wc = self.init_weights(
                self.input_dim, self.hidden_dim, name="Cell_Hidden_weight", reg=None
            )
            self.Uc = self.init_weights(
                self.hidden_dim, self.hidden_dim, name="Cell_State_weight", reg=None
            )
            self.bc = self.init_bias(self.hidden_dim, name="Cell_Hidden_bias")

            self.W_decomp = self.init_weights(
                self.hidden_dim,
                self.hidden_dim,
                name="Decomposition_Hidden_weight",
                reg=None,
            )
            self.b_decomp = self.init_bias(
                self.hidden_dim, name="Decomposition_Hidden_bias_enc"
            )

            self.Wo = self.init_weights(
                self.hidden_dim, fc_dim, name="Fc_Layer_weight", reg=None
            )  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.bo = self.init_bias(fc_dim, name="Fc_Layer_bias")

            self.W_softmax = self.init_weights(
                fc_dim, output_dim, name="Output_Layer_weight", reg=None
            )  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.b_softmax = self.init_bias(output_dim, name="Output_Layer_bias")

        else:
            self.Wi = self.no_init_weights(
                self.input_dim, self.hidden_dim, name="Input_Hidden_weight"
            )
            self.Ui = self.no_init_weights(
                self.hidden_dim, self.hidden_dim, name="Input_State_weight"
            )
            self.bi = self.no_init_bias(self.hidden_dim, name="Input_Hidden_bias")

            self.Wf = self.no_init_weights(
                self.input_dim, self.hidden_dim, name="Forget_Hidden_weight"
            )
            self.Uf = self.no_init_weights(
                self.hidden_dim, self.hidden_dim, name="Forget_State_weight"
            )
            self.bf = self.no_init_bias(self.hidden_dim, name="Forget_Hidden_bias")

            self.Wog = self.no_init_weights(
                self.input_dim, self.hidden_dim, name="Output_Hidden_weight"
            )
            self.Uog = self.no_init_weights(
                self.hidden_dim, self.hidden_dim, name="Output_State_weight"
            )
            self.bog = self.no_init_bias(self.hidden_dim, name="Output_Hidden_bias")

            self.Wc = self.no_init_weights(
                self.input_dim, self.hidden_dim, name="Cell_Hidden_weight"
            )
            self.Uc = self.no_init_weights(
                self.hidden_dim, self.hidden_dim, name="Cell_State_weight"
            )
            self.bc = self.no_init_bias(self.hidden_dim, name="Cell_Hidden_bias")

            self.W_decomp = self.no_init_weights(
                self.hidden_dim, self.hidden_dim, name="Decomposition_Hidden_weight"
            )
            self.b_decomp = self.no_init_bias(
                self.hidden_dim, name="Decomposition_Hidden_bias_enc"
            )

            self.Wo = self.no_init_weights(
                self.hidden_dim, fc_dim, name="Fc_Layer_weight"
            )
            self.bo = self.no_init_bias(fc_dim, name="Fc_Layer_bias")

            self.W_softmax = self.no_init_weights(
                fc_dim, output_dim, name="Output_Layer_weight"
            )
            self.b_softmax = self.no_init_bias(output_dim, name="Output_Layer_bias")

    def TLSTM_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])

        # Dealing with time irregularity

        # Map elapse time in days or months
        T = self.map_elapse_time(t)

        # Decompose the previous cell if there is a elapse time
        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Input gate
        i = tf.sigmoid(
            tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi
        )

        # Forget Gate
        f = tf.sigmoid(
            tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf
        )

        # Output Gate
        o = tf.sigmoid(
            tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog
        )

        # Candidate Memory Cell
        C = tf.nn.tanh(
            tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc
        )

        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    def get_states(self):  # Returns all hidden states for the samples in a batch
        batch_size = tf.shape(self.input)[0]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input = tf.transpose(
            scan_input_
        )  # scan input is [seq_length x batch_size x input_dim]
        scan_time = tf.transpose(self.time)  # scan_time [seq_length x batch_size]
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(
            scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1]
        )
        concat_input = tf.concat(
            [scan_time, scan_input], 2
        )  # [seq_length x batch_size x input_dim+1]
        packed_hidden_states = tf.scan(
            self.TLSTM_Unit, concat_input, initializer=ini_state_cell, name="states"
        )
        all_states = packed_hidden_states[:, 0, :, :]
        return all_states

    def get_output(self, state):
        output = tf.nn.relu(tf.matmul(state, self.Wo) + self.bo)
        output = tf.nn.dropout(output, self.keep_prob)
        output = tf.matmul(output, self.W_softmax) + self.b_softmax
        return output

    def get_outputs(self):  # Returns all the outputs
        all_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_states)
        output = tf.reverse(all_outputs, [0])[
            0, :, :
        ]  # Compatible with tensorflow 1.2.1
        # output = tf.reverse(all_outputs, [True, False, False])[0, :, :] # Compatible with tensorflow 0.12.1
        return output

    def get_cost_acc(self):
        logits = self.get_outputs()
        cross_entropy = tf.math.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
        )
        y_pred = tf.math.argmax(logits, 1)
        y = tf.math.argmax(self.labels, 1)
        return cross_entropy, y_pred, y, logits, self.labels

    def map_elapse_time(self, t):
        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)

        # T = tf.multiply(self.wt, t) + self.bt

        T = tf.math.divide(c1, tf.math.log(t + c2), name="Log_elapse_time")

        Ones = tf.ones([1, self.hidden_dim], dtype=tf.float32)

        T = tf.linalg.matmul(T, Ones)

        return T
