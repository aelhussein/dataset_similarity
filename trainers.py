ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl'
import copy
import torch
import torch.nn as nn
import sys
import numpy as np
sys.path.append(f'{ROOT_DIR}/code/run_models')
from sklearn import metrics
from torch.utils.data  import DataLoader
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from helper import *



@dataclass
class TrainerConfig:
    """Configuration for training parameters."""
    dataset_name: str
    device: str
    learning_rate: float
    batch_size: int
    epochs: int = 5
    rounds: int = 20
    personalization_params: Optional[Dict] = None


@dataclass
class SiteData:
    """Holds DataLoader and metadata for a site."""
    site_id: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    weight: float = 1.0
    
    def __post_init__(self):
        if self.train_loader is not None:
            self.num_samples = len(self.train_loader.dataset)

@dataclass
class ModelState:
    """Holds state for a single model (global or personalized)."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    best_loss: float = float('inf')
    best_model: Optional[nn.Module] = None
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_scores: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)
    test_scores: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.best_model is None and self.model is not None:
            self.best_model = copy.deepcopy(self.model)
    
    def copy(self):
        """Create a new ModelState with copied model and optimizer."""
        # Create new model instance
        new_model = copy.deepcopy(self.model)
        
        # Setup optimizer
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        new_optimizer = type(self.optimizer)(new_model.parameters(), **self.optimizer.defaults)
        new_optimizer.load_state_dict(optimizer_state)
        
        # Create new model state
        return ModelState(
            model=new_model,
            optimizer=new_optimizer,
            criterion= self.criterion 
        )

@dataclass
class SiteTrainerState:
    """Holds training-related state for a site."""
    global_state: ModelState
    personal_state: Optional[ModelState] = None

class Site:
    """Combines site data and training state."""
    def __init__(self, config: TrainerConfig, data: SiteData, state: SiteTrainerState):
        self.config = config
        self.data = data
        self.state = state

class ModelManager:
    """Manages model operations for a site."""
    def __init__(self, site: Site):
        self.site = site
        self.device = site.config.device

    def get_model_state(self, personal  = False):
        """Get model state (global or personal)."""
        state = self.site.state.personal_state if personal else self.site.state.global_state
        return state.model.state_dict()

    def set_model_state(self, state_dict, personal = False):
        """Set model state (global or personal)."""
        state = self.site.state.personal_state if personal else self.site.state.global_state
        state.model.load_state_dict(state_dict)


    def update_best_model(self, loss, personal = False):
        """Update the best model if loss improves."""
        state = self.site.state.personal_state if personal else self.site.state.global_state
        if loss < state.best_loss:
            state.best_loss = loss
            state.best_model = copy.deepcopy(state.model)
            return True
        return False
    
class MetricsCalculator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.continuous_outcome = ['Weather']
        self.squeeze_required = ['Synthetic', 'Credit']
        self.long_required = ['CIFAR', 'EMNIST', 'ISIC']
        self.tensor_metrics = ['IXITiny']
        
    def get_metric_function(self):
        """Returns appropriate metric function based on dataset."""
        metric_mapping = {
            'Synthetic': metrics.f1_score,
            'Credit': metrics.f1_score,
            'Weather': metrics.r2_score,
            'EMNIST': metrics.accuracy_score,
            'CIFAR': metrics.accuracy_score,
            'IXITiny': get_dice_score,
            'ISIC': metrics.balanced_accuracy_score
        }
        return metric_mapping[self.dataset_name]

    def process_predictions(self, predictions, labels):
        """Process model predictions based on dataset requirements."""
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        if self.dataset_name in self.continuous_outcome:
            predictions = np.clip(predictions, -2, 2)
        elif self.dataset_name in self.squeeze_required:
            predictions = (predictions >= 0.5).astype(int)
        elif self.dataset_name in self.long_required:
            predictions = predictions.argmax(axis=1)
            
        return predictions, labels

    def calculate_score(self, true_labels, predictions):
        """Calculate appropriate metric score."""
        metric_func = self.get_metric_function()
        
        if self.dataset_name in self.tensor_metrics:
            return metric_func(
                torch.tensor(true_labels, dtype=torch.float32),
                torch.tensor(predictions, dtype=torch.float32)
            )
        return metric_func(true_labels, predictions)


class Client(ModelManager):
    """Adds training capabilities to model management."""
    def __init__(self, site: Site, metrics_calculator: MetricsCalculator):
        super().__init__(site)
        self.metrics_calculator = metrics_calculator

    def train_epoch(self, model, state):
        """Train for one epoch using the provided model (already on correct device)."""
        total_loss = 0
        for batch_x, batch_y in self.site.data.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            state.optimizer.zero_grad()
            outputs = model(batch_x)
            loss = state.criterion(outputs, batch_y)
            loss.backward()
            state.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.site.data.train_loader)
        state.train_losses.append(avg_loss)
        return avg_loss

    def train(self, personal=False):
        """Train for multiple epochs using the specified model."""
        state = self.site.state.personal_state if personal else self.site.state.global_state
        model = state.model.train().to(self.device)
        try:
            for epoch in range(self.site.config.epochs):
                epoch_loss = self.train_epoch(model, state)
        finally:
            model.to('cpu')
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
        return epoch_loss # server only sent the loss from final epoch

    def evaluate(self, loader, personal= False, validate = False):
        """Evaluate specified model."""
        state = self.site.state.personal_state if personal else self.site.state.global_state
        if validate:
            model = state.model.to(self.device)
        else:
            model = state.best_model.to(self.device)
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = model(batch_x)
                loss = state.criterion(outputs, batch_y)
                total_loss += loss.item()
                predictions, labels = self.metrics_calculator.process_predictions(outputs, batch_y)
                all_predictions.extend(predictions)
                all_labels.extend(labels)


        avg_loss = total_loss / len(loader)
        score = self.metrics_calculator.calculate_score(np.array(all_labels), np.array(all_predictions))
        model.to('cpu')
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return avg_loss, score

    def validate(self, personal=False):
        """Validate the current model and update best model if improved."""
        state = self.site.state.personal_state if personal else self.site.state.global_state
        val_loss, val_score = self.evaluate(self.site.data.val_loader, personal, validate = True)
        
        # Store the validation loss
        state.val_losses.append(val_loss)
        
        # Update best model if validation loss improved
        if val_loss < state.best_loss:
            state.best_loss = val_loss
            state.best_model = copy.deepcopy(state.model)
        
        return val_loss, val_score

    def test(self, personal=False):
        """Test using the best model."""
        #Test
        test_loss, test_score = self.evaluate(self.site.data.test_loader, personal)
        
        state = self.site.state.personal_state if personal else self.site.state.global_state
        # Store the test metrics
        state.test_losses.append(test_loss)
        state.test_scores.append(test_score)
        
        return test_loss, test_score

class PFedMeClient(Client):
    def train(self, personal = True):
        """Train personal model with proximal term regularization."""
        state = self.site.state.personal_state
        model = state.model.train().to(self.device)
        global_model = self.site.state.global_state.model
        global_model = global_model.to(self.device)
        reg_param = self.site.config.personalization_params['reg_param']
        try:
            for epoch in range(self.site.config.epochs):
                # Single training step with proximal term
                epoch_loss = self.train_epoch(model, state)
                # Add proximal term to gradients
                for param_p, param_g in zip(model.parameters(), global_model.parameters()):
                    if param_p.grad is not None:
                        proximal_term = reg_param * (param_p - param_g)
                        param_p.grad.add_(proximal_term)
                        
                state.optimizer.step()
        finally:
            model.to('cpu')
            global_model.to('cpu')
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        return epoch_loss  # server only sent the loss from final epoch

class DittoClient(Client):
    def train(self, personal=False):
        """Train both global and personal models for Ditto."""
        # First train global model
        global_loss = super().train(personal=False)
        
        # Then train personal model with regularization
        state = self.site.state.personal_state
        reg_param = self.site.config.personalization_params['reg_param']
        model = state.model.train().to(self.device)
        global_model = self.site.state.global_state.model.to(self.device)
        
        try:
            # Train personal model
            personal_loss = 0
            for epoch in range(self.site.config.epochs):
                personal_loss = self.train_epoch(model, state)
                
                # Add regularization toward global model
                for param_p, param_g in zip(model.parameters(), global_model.parameters()):
                    if param_p.grad is not None:
                        reg_term = reg_param * (param_p - param_g)
                        param_p.grad.add_(reg_term)
                        
                state.optimizer.step()
                
        finally:
            model.to('cpu')
            global_model.to('cpu')
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        return personal_loss 
    
class Server:
    """Coordinates federated learning across sites."""
    def __init__(self, config: TrainerConfig, globalmodelstate: ModelState):
        self.device = config.device
        self.config = config
        self.clients = {}
        self.global_site = Site(
            config = self.config,
            data=SiteData(
                site_id='global',
                train_loader=None,
                val_loader=None,
                test_loader=None,
                weight=0
            ),
            state=SiteTrainerState(
                global_state = globalmodelstate,
            )
        )
        self.metrics_calculator = MetricsCalculator(self.config.dataset_name)

    def set_server_type(self, name):
        self.server_type = name

    def _create_client(self, site: Site):
        """Create a client instance - can be overridden by subclasses."""
        return Client(site=site, metrics_calculator=self.metrics_calculator)

    def add_client(self, clientdata: SiteData, personal: bool = False):
        """Add a client to the federation."""        
        # Create client state
        global_state = self.global_site.state.global_state
        clientstate = SiteTrainerState(
            global_state=global_state.copy(),
            personal_state=global_state.copy() if personal else None
        )
    
        # Create client
        site = Site(
            config = self.config,
            data=clientdata,
            state=clientstate
        )

        # Create client
        client = self._create_client(site)
        
        # Add client to federation
        self.clients[clientdata.site_id] = client
        self._update_site_weights()

    def _update_site_weights(self):
        """Update site weights based on dataset sizes."""
        total_samples = sum(client.site.data.num_samples for client in self.clients.values())
        for client in self.clients.values():
            client.site.data.weight = client.site.data.num_samples / total_samples

    def distribute_global_model(self):
        """Base distribution method - to be implemented by subclasses"""
        return
    
    def aggregate_models(self):
        """Base aggregation method - to be implemented by subclasses."""
        return

    def train_round(self):
        """Run one round of training."""
        global_state = self.global_site.state.global_state

        global_train_loss = 0
        global_val_loss = 0
        global_val_score = 0
        for round in range(self.config.rounds):
            for client in self.clients.values():
                weight = client.site.data.weight
                # Train and validate
                train_loss = client.train()
                val_loss, val_score = client.validate()
                #Weight metrics per site weight
                global_train_loss += train_loss * weight
                global_val_loss += val_loss * weight
                global_val_score += val_score * weight
            # Add to the global metrics tracking
            global_state.train_losses.append(global_train_loss)
            global_state.val_losses.append(global_val_loss)
            global_state.val_scores.append(global_val_score)

        self.aggregate_models()
        self.distribute_global_model()

        # If val loss seen by server ie end of the round improves performance - save model as best
        if global_val_loss < global_state.best_loss:
            global_state.best_loss = global_val_loss
            global_state.best_model = copy.deepcopy(global_state.model)

        return
    
    def test_global(self):
        """Test the global model across all clients."""
        global_state = self.global_site.state.global_state
        avg_test_loss = 0
        avg_test_score = 0

        for client in self.clients.values():
            test_loss, test_score = client.test()
            avg_test_loss += test_loss * client.site.data.weight
            avg_test_score += test_score * client.site.data.weight

        global_state.test_losses.append(avg_test_loss)
        global_state.test_scores.append(avg_test_score)

        return avg_test_loss, avg_test_score
    
    def calculate_diversity(self, client_id_1, client_id_2):
        """
        Calculate gradient diversity and weight divergence between two clients.
        """
        client_1 = self.clients[client_id_1]
        client_2 = self.clients[client_id_2]
        diversity_calculator = ModelDiversity(client_1, client_2)

        grad_diversity = diversity_calculator.calculate_gradient_diversity()
        weight_div, weight_orient = diversity_calculator.calculate_weight_divergence()

        return {
            "gradient_diversity": grad_diversity,
            "weight_divergence": weight_div,
            "weight_orientation": weight_orient
        }

class FedAvgServer(Server):
    """Implements FedAvg aggregation."""
    def __init__(self, config: TrainerConfig, globalmodelstate: ModelState):
        super().__init__(config, globalmodelstate)
    
    def aggregate_models(self):
        """Standard FedAvg aggregation."""
        global_model = self.global_site.state.global_state.model
        
        # Iterate through each layer's parameters using enumerate
        for param_idx, global_param in enumerate(global_model.parameters()):
            weighted_sum = torch.zeros_like(global_param, dtype=torch.float32)
            
            # For each client, get the corresponding parameter at the same index
            for client in self.clients.values():
                client_params = list(client.site.state.global_state.model.parameters())
                weighted_sum += client.site.data.weight * client_params[param_idx]
                
            global_param.data.copy_(weighted_sum)
    
    def distribute_global_model(self):
        """Distribute the global model to all clients."""
        global_state_dict = self.global_site.state.global_state.model.state_dict()
        for client in self.clients.values():
            client.set_model_state(global_state_dict)


class PFedMeServer(FedAvgServer):
    def _create_client(self, site: Site):
        """Create a PFedMe client instance."""
        return PFedMeClient(site=site, metrics_calculator=self.metrics_calculator)

class DittoServer(FedAvgServer):
    """Implements Ditto with personalization."""
    def __init__(self, config: TrainerConfig, globalmodelstate: ModelState):
        super().__init__(config, globalmodelstate)
    
    def _create_client(self, site: Site):
        """Create a Ditto client instance."""
        return DittoClient(site=site, metrics_calculator=self.metrics_calculator)        

class ModelDiversity:
    """Calculates diversity metrics between two clients' models."""
    def __init__(self, client_1: Client, client_2: Client):
        self.client_1 = client_1
        self.client_2 = client_2

    def calculate_gradient_diversity(self):
        """Calculate gradient diversity between two clients."""
        grads_1 = self._get_gradients(self.client_1)
        grads_2 = self._get_gradients(self.client_2)
        numerator = torch.norm(grads_1)**2 + torch.norm(grads_2)**2
        denominator = torch.norm(grads_1 + grads_2)**2
        return (numerator / denominator).item() if denominator != 0 else 0

    def calculate_weight_divergence(self):
        """Calculate weight divergence metrics between two clients."""
        weights_1 = self._get_weights(self.client_1)
        weights_2 = self._get_weights(self.client_2)
        
        # Normalize weights
        norm_1 = torch.norm(weights_1)
        norm_2 = torch.norm(weights_2)
        w1_normalized = weights_1 / norm_1 if norm_1 != 0 else weights_1
        w2_normalized = weights_2 / norm_2 if norm_2 != 0 else weights_2
        
        # Calculate divergence metrics
        weight_div = torch.norm(w1_normalized - w2_normalized)
        weight_orient = torch.dot(w1_normalized, w2_normalized)
        
        return weight_div.item(), weight_orient.item()

    def _get_gradients(self, client: Client):
        """Extract gradients from a client's model."""
        gradients = []
        state = client.site.state.global_state
        for param in state.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))  # Flatten gradients
        return torch.cat(gradients)

    def _get_weights(self, client: Client):
        """Extract weights from a client's model."""
        weights = []
        state = client.site.state.global_state
        for param in state.model.parameters():
            weights.append(param.data.view(-1))  # Flatten weights
        return torch.cat(weights)



def get_dice_score(output, target, SPATIAL_DIMENSIONS = (2, 3, 4), epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(axis=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(axis=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(axis=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score.mean().item()

def get_soft_dice_metric(y_pred, y_true, SPATIAL_DIMENSIONS = (2, 3, 4), epsilon=1e-9):
    """
    Soft Dice coefficient
    """
    intersection = (y_pred * y_true).sum(axis=SPATIAL_DIMENSIONS)
    union = (0.5 * (y_pred + y_true)).sum(axis=SPATIAL_DIMENSIONS)
    dice = intersection / (union + epsilon)
    # If both inputs are empty the dice coefficient should be equal 1
    dice[union == 0] = 1
    return np.mean(dice)


