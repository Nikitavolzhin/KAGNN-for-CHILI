import warnings
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN, EdgeCNN, GIN
from dataset_class import CHILI
import argparse
import optuna
import gc
from regression_models import KAGCN as KAGCN_reg, KAGIN as KAGIN_reg, KAEdge as KAEdge_reg
from classification_models import KAGCN_cls, KAGIN_cls, KAEdge_cls
from torcheval.metrics.functional import multiclass_f1_score
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
from pathlib import Path
import yaml

def position_MAE(pred_xyz, true_xyz):
    """
    Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
    """
    return torch.mean(
        torch.sqrt(torch.sum(F.mse_loss(pred_xyz, true_xyz, reduction="none"), dim=1)),
        dim=0,
    )
def train(model_name, params, train_loader, val_loader, test_loader, task):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_node_features = 7
    # Initialise loss function and metric function
    if task in {"edge_attr_regression", "saxs_regression", "xrd_regression", "xPDF_regression", "pos_abs_regression"}:
        loss_function = nn.SmoothL1Loss()
        metric_function = position_MAE if task == "pos_abs_regression" else nn.MSELoss()
        if task == "edge_attr_regression":
            num_classes = 300
        elif task == "xrd_regression":
            num_classes = 580
        elif task == "xPDF_regression":
            num_classes = 6000
        elif task == "pos_abs_regression":
            num_classes = 3
            num_node_features = 4
        else:
            num_classes = 1
        improved_function = lambda best, new: new < best if best is not None else True
        if model_name == "KAGIN":
            model = KAGIN_reg(num_node_features, 1, params["num_layers"], params["hidden_channels"], 1,
                              params["grid_size"], params["spline_order"],
                              num_classes, params["dropout_rate"], False).to(device)
        elif model_name == "KAGCN":
            model = KAGCN_reg(num_node_features, params["num_layers"], params["hidden_channels"], params["grid_size"],
                              params["spline_order"], num_classes,
                              params["dropout_rate"], False).to(device)
        elif model_name == "KAEdge":
            model = KAEdge_reg(num_node_features, 1, params["num_layers"], params["hidden_channels"], 1,
                              params["grid_size"], params["spline_order"],
                              num_classes, params["dropout_rate"]).to(device)

    elif task in {"crystal_system_classification", "space_group_classification", "atom_classification"}:
        if task == "atom_classification":
            num_classes = 118
            num_node_features = 3
        else:
            num_node_features = 7
        loss_function = lambda x,y: cross_entropy(x, y.long() - 1)
        num_classes = 7 if task == "crystal_system_classification" else 230
        metric_function = lambda x,y: multiclass_f1_score(x, y.long() - 1, num_classes=num_classes, average='weighted')
        improved_function = lambda best, new: new > best if best is not None else True
        if model_name == "KAGIN":
            model = KAGIN_cls(params["num_layers"], num_node_features, params["hidden_channels"], num_classes, 1,
                              params["grid_size"], params["spline_order"],
                              params["dropout_rate"]).to(device)
        elif model_name == "KAGCN":
            model = KAGCN_cls(params["num_layers"], num_node_features, params["hidden_channels"], num_classes,
                              params["grid_size"], params["spline_order"],params["dropout_rate"]).to(device)
        elif  model_name == "KAEdge":
            model = KAEdge_cls(params["num_layers"], num_node_features, params["hidden_channels"], num_classes, 1,
                              params["grid_size"], params["spline_order"],
                              params["dropout_rate"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    # Training & Validation
    patience = 0
    best_error = None
    # try-catch block to catch Out of Memory issue
    try:
        for epoch in range(params["max_epochs"]):

            # Patience
            if patience >= params["max_patience"]:
                print("Max Patience reached, quitting...", flush=True)
                break

            # Training loop
            model.train()
            train_loss = 0
            for data in train_loader:
                # Send to device
                data = data.to(device)

                # Perform forward pass
                if task in {"saxs_regression", "xrd_regression", "xPDF_regression"}:
                    pred = model.forward(
                        x = torch.cat((data.x, data.pos_abs), dim=1),
                        edge_index = data.edge_index,
                        edge_attr = None,
                        edge_weight = None,
                        batch = data.batch,
                        node_level = False
                    )
                elif task == "pos_abs_regression":
                    pred = model.forward(
                        x = data.x,
                        edge_index = data.edge_index,
                        edge_attr = None,
                        edge_weight = None,
                        batch = data.batch,
                    )
                elif task == "atom_classification":
                    pred = model.forward(
                        x=data.pos_abs,
                        edge_index=data.edge_index,
                        edge_attr=None,
                        edge_weight=None,
                        batch=data.batch,
                        node_level=True
                    )
                elif task == "edge_attr_regression":
                    pred = model.forward(
                        x=torch.cat((data.x, data.pos_abs), dim=1),
                        edge_index=data.edge_index,
                        edge_attr=None,
                        edge_weight=None,
                        batch=data.batch,
                        node_level=True
                    )
                else:
                    pred = model.forward(
                        x = torch.cat((data.x, data.pos_abs), dim=1),
                        edge_index = data.edge_index,
                        edge_attr = None,
                        edge_weight = None,
                        batch = data.batch,
                        node_level=False
                    )

                if task == "edge_attr_regression":
                    pred = torch.sum(pred[data.edge_index[0, :]] * pred[data.edge_index[1, :]], dim = -1)
                    truth = data.edge_attr
                elif task == "crystal_system_classification":
                    truth = data.y['crystal_system_number']
                elif task == "space_group_classification":
                    truth = torch.tensor(data.y['space_group_number']).to(device)
                elif task == "saxs_regression":
                    truth = data.y['saxs'][1::2, :]
                    truth_min = torch.min(truth, dim=-1, keepdim=True)[0]
                    truth_max = torch.max(truth, dim=-1, keepdim=True)[0]
                    truth = (truth - truth_min) / (truth_max - truth_min)
                elif task == "xrd_regression":
                    truth = data.y['xrd'][1::2, :]
                    truth_min = torch.min(truth, dim=-1, keepdim=True)[0]
                    truth_max = torch.max(truth, dim=-1, keepdim=True)[0]
                    truth = (truth - truth_min) / (truth_max - truth_min)
                elif task == "xPDF_regression":
                    truth = data.y['xPDF'][1::2, :]
                    truth_min = torch.min(truth, dim=-1, keepdim=True)[0]
                    truth_max = torch.max(truth, dim=-1, keepdim=True)[0]
                    truth = (truth - truth_min) / (truth_max - truth_min)
                elif task == "pos_abs_regression":
                    truth = data.pos_abs
                elif task == "atom_classification":
                    truth = data.x[:, 0].long()

                loss = loss_function(pred.to(device), truth)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Training loss
            train_loss = train_loss / len(train_loader)

            # Validation loop
            model.eval()
            val_error = 0
            for data in val_loader:

                # Send to device
                data = data.to(device)

                # Perform forward pass
                with torch.no_grad():
                    if task in {"saxs_regression", "xrd_regression", "xPDF_regression"}:
                        pred = model.forward(
                            x=torch.cat((data.x, data.pos_abs), dim=1),
                            edge_index=data.edge_index,
                            edge_attr=None,
                            edge_weight=None,
                            batch=data.batch,
                            node_level=False
                        )

                    elif task == "pos_abs_regression":
                        pred = model.forward(
                            x=data.x,
                            edge_index=data.edge_index,
                            edge_attr=None,
                            edge_weight=None,
                            batch=data.batch,
                        )
                    elif task == "atom_classification":
                        pred = model.forward(
                            x=data.pos_abs,
                            edge_index=data.edge_index,
                            edge_attr=None,
                            edge_weight=None,
                            batch=data.batch,
                            node_level=True
                        )
                    elif task == "edge_attr_regression":
                        pred = model.forward(
                            x=torch.cat((data.x, data.pos_abs), dim=1),
                            edge_index=data.edge_index,
                            edge_attr=None,
                            edge_weight=None,
                            batch=data.batch,
                            node_level=True
                        )
                    else:
                        pred = model.forward(
                            x=torch.cat((data.x, data.pos_abs), dim=1),
                            edge_index=data.edge_index,
                            edge_attr=None,
                            edge_weight=None,
                            batch=data.batch,
                            node_level=False
                        )
                    if task == "edge_attr_regression":
                        pred = torch.sum(pred[data.edge_index[0, :]] * pred[data.edge_index[1, :]], dim = -1)
                        truth = data.edge_attr
                    elif task == 'crystal_system_classification':
                        truth = data.y['crystal_system_number']
                    elif task == "space_group_classification":
                        truth =  torch.tensor(data.y['space_group_number']).to(device)
                    elif task == "saxs_regression":
                        truth = data.y['saxs'][1::2, :]
                        truth_min = torch.min(truth, dim=-1, keepdim=True)[0]
                        truth_max = torch.max(truth, dim=-1, keepdim=True)[0]
                        truth = (truth - truth_min) / (truth_max - truth_min)
                    elif task == "xrd_regression":
                        truth = data.y['xrd'][1::2, :]
                        truth_min = torch.min(truth, dim=-1, keepdim=True)[0]
                        truth_max = torch.max(truth, dim=-1, keepdim=True)[0]
                        truth = (truth - truth_min) / (truth_max - truth_min)
                    elif task == "xPDF_regression":
                        truth = data.y['xPDF'][1::2, :]
                        truth_min = torch.min(truth, dim=-1, keepdim=True)[0]
                        truth_max = torch.max(truth, dim=-1, keepdim=True)[0]
                        truth = (truth - truth_min) / (truth_max - truth_min)
                    elif task == "pos_abs_regression":
                        truth = data.pos_abs
                    elif task == "atom_classification":
                        truth = data.x[:, 0].long()
                    metric = metric_function(pred.to(device), truth)

                # Aggregate errors
                val_error += metric.item()

            val_error = val_error / len(val_loader)

            if improved_function(best_error, val_error):
                best_error = val_error
                patience = 0
            else:
                patience += 1

            # Print checkpoint
            if task in {"edge_attr_regression", "saxs_regression", "xrd_regression", "xPDF_regression"}:
                print(f'Epoch: {epoch+1}/{params["max_epochs"]}, Train Loss: {train_loss:.4f}, Val MSE: {val_error:.4f}')
            elif task in {"crystal_system_classification", "space_group_classification", "atom_classification"}:
                print(f'Epoch: {epoch+1}/{params["max_epochs"]}, Train Loss: {train_loss:.4f}, Val weighted F1: {val_error:.4f}')
            elif task == "pos_abs_regression":
                print(f'Epoch: {epoch + 1}/{params["max_epochs"]}, Train Loss: {train_loss:.4f}, Val MAE: {val_error:.4f}')

        return val_error
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(">>> Caught OOM during training, clearing GPU and returning Inf loss.")
            torch.cuda.empty_cache()
            if task in {"crystal_system_classification", "space_group_classification", "atom_classification"}:
                return -1
            else:
                return float("inf")
        else:
            raise
# objective function for Optuna hyperparameter tuning
def objective_function(trial, model_name, train_loader, val_loader, test_loader, metrics_file, task):

    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    batch_size = 32
    hidden_channels = trial.suggest_int('hidden_channels', 16, 64)
    dropout_rate = trial.suggest_float('dropout_rate', 0., 0.5)
    params = {
        'learning_rate': lr,
        'num_layers': num_layers,
        'batch_size': batch_size,
        'hidden_channels': hidden_channels,
        'max_epochs': 500,
        'max_patience': 20,
        'dropout_rate': dropout_rate
    }
    if model_name in {"KAGIN", "KAGCN", "KAEdge"}:
        grid_size = trial.suggest_categorical('grid_size', [3, 4, 5])
        params['grid_size'] = grid_size
        spline_order = trial.suggest_categorical('spline_order', [3, 4, 5])
        params['spline_order'] = spline_order
    best_val_loss = train(model_name, params, train_loader, val_loader, test_loader, task)
    # to resolve the memory issue
    torch.cuda.empty_cache()
    gc.collect()

    metrics_file.write(f"Best validation loss: {best_val_loss}\nWith params: {params}\n")
    return best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment script")
    parser.add_argument("-d", "--config_dataset", type=str, help="CHILI Dataset configutaion 3 or 100 (K)")
    parser.add_argument("-m", "--model", type=str, help="type of the model")
    parser.add_argument("-t", "--task", type=str, help="type of the task")

    args = parser.parse_args()

    # Create dataset
    root = 'benchmark/dataset/'
    dataset = f'CHILI-{args.config_dataset}K'
    dataset = CHILI(root, dataset)
    print(f"Dataset version: {args.config_dataset}")
    # Create random split and load that into the dataset class
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dataset.create_data_split(split_strategy='random', test_size=0.1, stratify_on='crystal_system_number')
        dataset.load_data_split(split_strategy='random')

    batch_size = 32

    # Define dataloaders
    train_loader = DataLoader(dataset.train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset.validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset.test_set, batch_size=batch_size, shuffle=False)

    print(f"Number of training samples: {len(dataset.train_set)}", flush=True)
    print(f"Number of validation samples: {len(dataset.validation_set)}", flush=True)
    print(f"Number of test samples: {len(dataset.test_set)}", flush=True)

    model_name = args.model
    task = args.task
    with open(f"metrics_{args.config_dataset}K.txt", "a") as metrics_file:
        print("Model: " + model_name)
        metrics_file.write(f"Model: {model_name}\nTask: {task}\n")

        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(sampler=sampler, direction="maximize") if task in {"crystal_system_classification", "space_group_classification", "atom_classification"} else optuna.create_study(sampler=sampler)
        study.optimize(
            lambda trial: objective_function(trial, model_name, train_loader, val_loader, test_loader, metrics_file, task),
            n_trials=40,
            gc_after_trial=True)
        best_hyperparams = study.best_params
        print(best_hyperparams)
        metrics_file.write(f"Best hyperparameters: {best_hyperparams}\n")

    path = Path("parameters.yaml")
    if path.exists():
        with path.open('r') as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    best_hyperparams['batch_size'] = 32
    best_hyperparams['max_patience'] = 20
    best_hyperparams['max_epochs'] = 500
    if model_name not in data:
        data[model_name] = {}
    data[model_name][task] = best_hyperparams
    with path.open('w') as f:
        yaml.dump(data, f,
                  default_flow_style=False,
                  sort_keys=False)
