import os
os.environ["TF_MEMORY_ALLOCATION"] = "0.7"

from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
import nvtabular as nvt
import cudf

data_dir = "data/task_2.csv"


def get_ds_from_df(df_tmp, cat_cols, cont_cols, label=["overall"], batch_size=1024):
    # FIXME: add inputs to KerasSequenceLoader
    dataset = KerasSequenceLoader(
        nvt.Dataset(df_tmp),
        batch_size=batch_size,
        label_names=label,
        cat_names=cat_cols,
        cont_names=cont_cols,
        shuffle=True,
        buffer_size=0.06
    )
    return dataset


def get_test_and_train(cat_cols=[], cont_cols=[], df=None):
    # FIXME: use inputs to select training and validation datasets
    ratings = cudf.read_csv(data_dir) if df is None else df
    train_ds = ratings[~ratings["valid"]]
    train_ds = get_ds_from_df(train_ds, cat_cols, cont_cols)

    valid_ds = ratings[ratings["valid"]]
    valid_ds = get_ds_from_df(valid_ds, cat_cols, cont_cols)

    return train_ds, valid_ds
