import click
import mlflow
import numpy as np
import xarray as xr
from functools import reduce
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso, MultiTaskLasso
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def fit_and_evaluate_lasso(train_data, test_data, alpha=1.0, n_components=0.9):
    # set up
    N_train = train_data.shape[0]
    N_test = test_data.shape[0]
    pca = PCA(n_components=n_components)
    multi_lasso = MultiTaskLasso(alpha=alpha)
    pipeline = Pipeline([('pca', pca), ('lasso', multi_lasso)])
    train_data = train_data.values.reshape((N_train, train_data.shape[1]*train_data.shape[2]))
    train_x, train_y = train_data[:-1], train_data[1:]
    # fit model
    pipeline.fit(train_x, train_y)
    # evaluate
    test_data = test_data.values.reshape((N_test, test_data.shape[1]*test_data.shape[2]))
    test_x, test_y = test_data[:-1], test_data[1:]
    y_pred = pipeline.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_x, y_pred))
    mae = mean_absolute_error(test_x, y_pred)
    r2 = r2_score(test_x, y_pred)
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)

def fit_cv_lasso(train_data, test_data, fold_id, min_alpha, max_alpha, num_alpha):
    for alpha in np.geomspace(min_alpha, max_alpha, num_alpha):
        with mlflow.start_run(nested=True) as run:
            mlflow.log_param('fold', fold_id)
            mlflow.log_param('alpha', alpha)
            fit_and_evaluate_lasso(train_data, test_data, alpha=alpha)

@click.command(help="Fits and validates a multi-task LASSO model on the given NetCDF dataset")
@click.option("--min-alpha", type=click.FLOAT, default=1.0E-3, help="Minimum alpha parameter for lasso")
@click.option("--max-alpha", type=click.FLOAT, default=1.0E2, help="Maximum alpha parameter for lasso")
@click.option("--num-alpha", type=click.INT, default=10, help="Number of alpha parameters to fit")
@click.option("--num-splits", type=click.INT, default=5, help="Number of CV splits")
@click.option("--ds-var", type=click.STRING, default='tas', help="Dataset var name")
@click.argument("dataset_filename")
def run_cv(dataset_filename, min_alpha, max_alpha, num_alpha, num_splits, ds_var):
    dataset = xr.load_dataset(dataset_filename)[ds_var]
    time_split = TimeSeriesSplit(n_splits=num_splits)
    folds = list(time_split.split(dataset.time))
    for i, (train, test) in enumerate(folds):
        fit_cv_lasso(dataset[train], dataset[test], i+1, min_alpha, max_alpha, num_alpha)
        print('Finished fold {}/{}'.format(i+1, len(folds)))


if __name__ == '__main__':
    run_cv()