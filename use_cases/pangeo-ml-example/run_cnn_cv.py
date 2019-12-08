import click
import mlflow
import numpy as np
import xarray as xr
from itertools import product
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit

def fit_and_evaluate_cnn(train_data, test_data, num_layers=3, num_filters=16, kernel_size=3, l2_reg=1.0E-4):
    # set up
    train_data = np.expand_dims(train_data.values, axis=-1)
    test_data = np.expand_dims(test_data.values, axis=-1)
    input_0 = Input(shape=(None, None, 1))
    x = input_0
    for i in range(num_layers):
        mult = 2**i
        x = Conv2D(num_filters*mult, kernel_size, kernel_regularizer=l2(l2_reg), padding='same')(x)
        x = Activation('relu')(x)
    output_0 = Conv2D(1, 5, padding='same')(x)
    model = Model(inputs=input_0, outputs=output_0)
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_true - y_pred)))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', rmse])
    model.fit(train_data[:-1], train_data[1:], epochs=10, validation_data=(test_data[:-1], test_data[1:]),
              callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])
    _, mae, rmse = model.evaluate(test_data[:-1], test_data[1:])
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)

def fit_cv_cnn(train_data, test_data, fold_id, min_l2_reg, max_l2_reg, num_l2_reg, min_layers, max_layers):
    l2_params = np.geomspace(min_l2_reg, max_l2_reg, num_l2_reg)
    layers_params = list(range(min_layers, max_layers+1))
    for l2_reg, num_layers in product(l2_params, layers_params):
        with mlflow.start_run(nested=True) as run:
            mlflow.log_param('fold', fold_id)
            mlflow.log_param('l2', l2_reg)
            mlflow.log_param('num_layers', num_layers) 
            fit_and_evaluate_cnn(train_data, test_data, num_layers, l2_reg=l2_reg)

@click.command(help="Fits and validates a deep convolutional model on the given NetCDF dataset")
@click.option("--min-l2-reg", type=click.FLOAT, default=1.0E-3, help="Minimum l2 regularization parameter")
@click.option("--max-l2-reg", type=click.FLOAT, default=1.0E2, help="Maximum l2 regularization parameter")
@click.option("--num-l2-reg", type=click.INT, default=10, help="Number of l2 regularization parameters to fit")
@click.option("--min-layers", type=click.INT, default=1, help="Minimum number of layers to use")
@click.option("--max-layers", type=click.INT, default=4, help="Maximum number of layers to use")
@click.option("--num-splits", type=click.INT, default=5, help="Number of splits to use")
@click.option("--ds-var", type=click.STRING, default='tas', help="Dataset var name")
@click.argument("dataset_filename")
def run_cv(dataset_filename, min_l2_reg, max_l2_reg, num_l2_reg, min_layers, max_layers, num_splits, ds_var):
    mlflow.log_param('has_gpu', tf.test.is_gpu_available())
    dataset = xr.load_dataset(dataset_filename)[ds_var]
    time_split = TimeSeriesSplit(n_splits=num_splits)
    folds = list(time_split.split(dataset.time))
    for i, (train, test) in enumerate(folds):
        fit_cv_cnn(dataset[train], dataset[test], i+1, min_l2_reg, max_l2_reg, num_l2_reg, min_layers, max_layers)
        print('Finished fold {}/{}'.format(i+1, len(folds)))


if __name__ == '__main__':
    run_cv()