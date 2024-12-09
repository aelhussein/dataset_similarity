ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl'

import sys
import os
import numpy as np
import torch
import torch.nn as nn
sys.path.append(f'{ROOT_DIR}/code/run_models')
from data_preprocessing import *
from trainers import *
from helper import *
from hyperparameters import *
from losses import *
import models as ms
import pickle


class ExperimentType:
    LEARNING_RATE = 'learning_rate'
    REG_PARAM = 'reg_param'
    EVALUATION = 'evaluation'


class ExperimentConfig:
    def __init__(self, dataset, experiment_type, params_to_try=None):
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.params_to_try = params_to_try or self._get_params_test()

    def _get_params_test(self):
        if self.experiment_type == ExperimentType.LEARNING_RATE:
            return LEARNING_RATES_TRY
        elif self.experiment_type == ExperimentType.REG_PARAM:
            return REG_PARAMS_TRY
        else:
            return None

class ResultsManager:
    def __init__(self, root_dir, dataset, experiment_type):
        self.root_dir = root_dir
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.results_structure = {
            ExperimentType.LEARNING_RATE: {
                'directory': 'lr_tuning',
                'filename_template': f'{dataset}_lr_tuning.pkl',
            },
            ExperimentType.REG_PARAM: {
                'directory': 'reg_param_tuning',
                'filename_template': f'{dataset}_reg_tuning.pkl',
            },
            ExperimentType.EVALUATION: {
                'directory': 'evaluation',
                'filename_template': f'{dataset}_evaluation.pkl'
            }
        }
        self.base_dir = self._setup_results_directories()

    def _setup_results_directories(self):
        experiment_info = self.results_structure[self.experiment_type]
        base_dir = os.path.join(self.root_dir, 'results', experiment_info['directory'])
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def _get_results_path(self):
        experiment_info = self.results_structure[self.experiment_type]
        filename = experiment_info['filename_template']
        return os.path.join(self.base_dir, filename)

    def load_results(self):
        path = self._get_results_path(self.experiment_type)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def save_results(self, results):
        path = self._get_results_path()
        with open(path, 'wb') as f:
            pickle.dump(results, f)

    def append_or_create_metric_lists(self, existing_dict, new_dict):
        if existing_dict is None:
            return {k: [v] if not isinstance(v, dict) else 
                   self.append_or_create_metric_lists(None, v)
                   for k, v in new_dict.items()}
        
        for key, new_value in new_dict.items():
            if isinstance(new_value, dict):
                if key not in existing_dict:
                    existing_dict[key] = {}
                existing_dict[key] = self.append_or_create_metric_lists(
                    existing_dict[key], new_value)
            else:
                if key not in existing_dict:
                    existing_dict[key] = []
                existing_dict[key].append(new_value)
        
        return existing_dict

    def get_best_parameters(self,server_type, cost):
        results = self.load_results()
        if results is None or cost not in results:
            return None
        
        cost_results = results[cost]
        if server_type not in cost_results:
            return None

        return self._select_best_hyperparameter(cost_results[server_type])

    def _select_best_hyperparameter(self, server_results):
        best_loss = float('inf')
        best_param = None
        
        for param_value, metrics in server_results.items():
            # Handle case where we have multiple runs
            if isinstance(metrics['global']['losses'], list):
                avg_loss = np.mean([np.mean(run_losses) 
                                  for run_losses in metrics['global']['losses']])
            else:
                avg_loss = np.mean(metrics['global']['losses'])
                
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_param = param_value
                
        return best_param

    def aggregate_metrics(self, results):
        if results is None:
            return None

        def aggregate_leaf(values):
            if not isinstance(values, list):
                return values
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        def recursive_aggregate(d):
            if not isinstance(d, dict):
                return aggregate_leaf(d)
            return {k: recursive_aggregate(v) for k, v in d.items()}

        return recursive_aggregate(results)


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_manager = ResultsManager(root_dir=ROOT_DIR, dataset=self.config.dataset, experiment_type = self.config.experiment_type)
        self.data_dir, self.default_params = get_parameters_for_dataset(self.config.dataset)

    def run_experiment(self, costs):
        if self.config.experiment_type == ExperimentType.EVALUATION:
            return self._run_final_evaluation(costs)
        else:
            return self._run_hyperparameter_tuning(costs)
    
    def _run_hyperparameter_tuning(self, costs):
        """Run LR or Reg param tuning with multiple runs"""
        results = None
        for run in range(self.default_params['runs']):
            try:
                print(f"Starting run {run + 1}/{self.default_params['runs']}")
                results_run = {}
                
                for cost in costs:
                    if self.config.experiment_type == ExperimentType.LEARNING_RATE:
                        hyperparams_list = [{'learning_rate': lr} for lr in self.config.params_to_try]
                        server_types = ['single', 'joint', 'fedavg', 'pfedme', 'ditto']
                    else:  # REG_PARAM
                        hyperparams_list = [{'reg_param': reg} for reg in self.config.params_to_try]
                        server_types = ['pfedme', 'ditto']

                    tracking = {}
                    for hyperparams in hyperparams_list:
                        tracking_for_params = self._hyperparameter_tuning(cost, hyperparams, server_types)
                        tracking.update(tracking_for_params)
                    results_run[cost] = tracking
                
                results = self.results_manager.append_or_create_metric_lists(results, results_run)
                self.results_manager.save_results(results, self.config.experiment_type)
                
            except Exception as e:
                print(f"Run {run + 1} failed with error: {e}")
                if results is not None:
                    self.results_manager.save_results(results, self.config.experiment_type)
        
        return results
    
    def _hyperparameter_tuning(self, cost, hyperparams, server_types):
        """Run hyperparameter tuning for specific parameters."""
        client_dataloaders = self._initialize_experiment(self.default_params['batch_size'], cost)
        tracking = {}
        
        for server_type in server_types:
            print(f"Training {server_type} model with hyperparameters: {hyperparams}")
            lr = hyperparams.get('learning_rate', get_default_lr(self.config.dataset))
            
            if server_type in ['pfedme', 'ditto']:
                reg_param = hyperparams.get('reg_param', get_default_reg(self.config.dataset))
                config = self._create_trainer_config(lr, personalization_params={"reg_param": reg_param})
            else:
                config = self._create_trainer_config(lr)

            server = self._create_server_instance(server_type, config, cost)
            self._add_clients_to_server(server, client_dataloaders)
            metrics = self._train_and_evaluate(server, config.rounds, tuning=True)

            # Update tracking using the parameter being tuned
            param_value = hyperparams.get('learning_rate', hyperparams.get('reg_param'))
            if server_type not in tracking:
                tracking[server_type] = {}
            tracking[server_type][param_value] = metrics
            
        return tracking

    def _run_final_evaluation(self, costs):
        """Run final evaluation with multiple runs"""
        results = None
        
        for run in range(self.default_params['runs']):
            try:
                print(f"Starting run {run + 1}/{self.default_params['runs']}")
                results_run = {}
                
                for cost in costs:
                    experiment_results = self._final_evaluation(cost)
                    results_run[cost] = experiment_results
                
                results = self.results_manager.append_or_create_metric_lists(results, results_run)
                self.results_manager.save_results(results, self.config.experiment_type)
                
            except Exception as e:
                print(f"Run {run + 1} failed with error: {e}")
                if results is not None:
                    self.results_manager.save_results(results, self.config.experiment_type)
        
        return results


    def _final_evaluation(self, cost):
        tracking = {}
        server_types = ['single', 'joint', 'fedavg', 'pfedme', 'ditto']
        client_dataloaders = self._initialize_experiment(self.default_params['batch_size'], cost)

        for server_type in server_types:
            print(f"Evaluating {server_type} model with best hyperparameters")
            lr = self.results_manager.get_best_parameters(
                ExperimentType.LEARNING_RATE, server_type, cost)
            
            if server_type in ['pfedme', 'ditto']:
                reg_param = self.results_manager.get_best_parameters(
                    ExperimentType.REG_PARAM, server_type, cost)
                config = self._create_trainer_config(lr, personalization_params={"reg_param": reg_param})
            else:
                config = self._create_trainer_config(lr)

            server = self._create_server_instance(server_type, config, cost)
            self._add_clients_to_server(server, client_dataloaders)
            metrics = self._train_and_evaluate(server, config.rounds)
            tracking[server_type] = metrics

        return tracking
    
    
    def _initialize_experiment(self, batch_size, cost):
        preprocessor = DataPreprocessor(self.config.dataset, batch_size)
        client_data = {}
        client_ids = self._get_client_ids(cost)
        
        for client_id in client_ids:
            client_num = int(client_id.split('_')[1])
            X, y = self._load_data(client_num, cost)
            client_data[client_id] = {'X': X, 'y': y}
        
        return preprocessor.process_clients(client_data)
    
    def _get_client_ids(self, cost):
        CLIENT_NUMS = {'IXITiny': 3, 'ISIC': 4}
        if self.config.dataset in CLIENT_NUMS and cost == 'all':
            CLIENT_NUM = CLIENT_NUMS[self.config.dataset]
        else:
            CLIENT_NUM = 2
        return [f'client_{i}' for i in range(1, CLIENT_NUM + 1)]
    
    def _create_trainer_config(self, learning_rate, personalization_params = None):
        return TrainerConfig(
            dataset_name=self.config.dataset,
            device=DEVICE,
            learning_rate=learning_rate,
            batch_size=self.default_params['batch_size'],
            epochs=5,
            rounds=self.default_params['rounds'],
            personalization_params=personalization_params
        )

    def _create_model(self, architecture, cost, learning_rate):
        model = getattr(ms, self.config.dataset)()
        if self.config.dataset in ['EMNIST', 'CIFAR']:
            with open(f'{self.data_dir}/CLASSES', 'rb') as f:
                classes_used = pickle.load(f)
            classes = len(classes_used[cost][0]) if architecture == 'single' else len(
                set(classes_used[cost][0] + classes_used[cost][1]))
            model = getattr(ms, self.config.dataset)(classes)

        criterion = {
            'Synthetic': nn.BCELoss(),
            'Credit': nn.BCELoss(),
            'Weather': nn.MSELoss(),
            'EMNIST': nn.CrossEntropyLoss(),
            'CIFAR': nn.CrossEntropyLoss(),
            'IXITiny': get_dice_loss,
            'ISIC': nn.CrossEntropyLoss()
        }.get(self.config.dataset, None)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            amsgrad=True,
            weight_decay=1e-4
        )
        return model, criterion, optimizer

    def _create_server_instance(self, server_type, config, cost):
        learning_rate = config.learning_rate
        model, criterion, optimizer = self._create_model(server_type, cost, learning_rate)
        globalmodelstate = ModelState(
            model=model,
            optimizer=optimizer,
            criterion=criterion
        )

        server_mapping = {
            'single': Server,
            'joint': Server,
            'fedavg': FedAvgServer,
            'pfedme': PFedMeServer,
            'ditto': DittoServer
        }

        server_class = server_mapping[server_type]
        server = server_class(config=config, globalmodelstate=globalmodelstate)
        server.set_server_type(server_type)
        return server

    def _add_clients_to_server(self, server, client_dataloaders):
        is_personalized = server.server_type in ['pfedme', 'ditto']
        for client_id in client_dataloaders:
                clientdata = self._create_site_data(client_id, client_dataloaders[client_id])
                server.add_client(clientdata=clientdata, personal=is_personalized)

    def _create_site_data(self, client_id, loaders):
        return SiteData(
            site_id=client_id,
            train_loader=loaders[0],
            val_loader=loaders[0],
            test_loader=loaders[0]
        )

    def _load_data(self, client_num, cost):
        return loadData(self.config.dataset, f'{self.data_dir}', client_num, cost)

    def _train_and_evaluate(self, server, rounds, tuning = False):
        for _ in range(rounds):
            server.train_round()
        # Collect metrics
        server.test_global()
        state = server.global_site.state.global_state
        if tuning:
            losses, scores = state.val_losses, state.val_scores 
        else:
            losses, scores = state.test_losses, state.test_scores 
        metrics = {
            'global': {
                'losses': losses,
                'scores': scores
            },
            'sites': {}
        }
        for client_id, client in server.clients.items():
            state = client.site.state.personal_state if client.site.state.personal_state is not None  else client.site.state.global_state
            if tuning:
                losses, scores = state.val_losses, state.val_scores 
            else:
                losses, scores = state.test_losses, state.test_scores 
            metrics['sites'][client_id] = {
                'losses': losses,
                'scores': scores
            }
        return metrics
