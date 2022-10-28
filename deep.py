import cudf
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, concatenate, Dense

ratings = cudf.read_csv("data/task_2.csv")


def get_model(
    embedding_inputs={},
    concatenate_layers=[],  # Ignore for now
    emb_size=32,
    hidden_layers=[256, 128],
):

    # Make embeddings
    embeddings = []
    for col in embedding_inputs:
        # FIXME: build a list of embedding layers
        max_index = int(ratings[col].max() + 1)
        embedding_input = embedding_inputs[col]
        embedding_layer = Embedding(max_index, emb_size, name="emb_" + col)
        embeddings.append(tf.squeeze(embedding_layer(embedding_input), axis=1))

    # The current layer at the end of our function chain
    end_layer = concatenate(concatenate_layers + embeddings, axis=1)
    for i, units in enumerate(hidden_layers):
        end_layer = Dense(units, activation="relu", name="dnn_{}".format(i))(end_layer)
    return Dense(1, activation=None, name="pred")(end_layer)
