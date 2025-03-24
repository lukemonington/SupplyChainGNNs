import torch
import numpy as np
import os
from flask import Flask, request, jsonify
import pickle
from torch_geometric.data import Data
import logging
import torch.serialization

# Add safe globals for loading edge_attr data
torch.serialization.add_safe_globals([
    "torch_geometric.data.data.DataEdgeAttr",  # Add any other classes that might be needed
])

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)

# Global variables to store models and data
models = {}
node_mapping = {}
edge_index = None

# MLP model class definition (ensure it matches your training implementation)
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# GCN model class definition
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GCN, self).__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# GAT model class definition - FIXED to match your training code
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=6, dropout_rate=0.5):
        super(GAT, self).__init__()
        from torch_geometric.nn import GATConv
        # GAT layer with multi-head attention (4 neurons per head)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        # Output layer (combines all heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# TGAT model class definition
class TGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, time_steps, output_dim, heads=6, dropout_rate=0.5):
        super(TGAT, self).__init__()
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv
        
        # Temporal embedding layer
        self.time_encoder = torch.nn.Linear(1, hidden_dim)
        
        # Node feature encoder
        self.feature_encoder = torch.nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)
        
        # Temporal attention (attention over time steps)
        self.temporal_attention = torch.nn.Parameter(torch.ones(time_steps))
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.time_steps = time_steps
        self.hidden_dim = hidden_dim
        self.heads = heads
    
    def forward(self, x, edge_index, time_idx=None):
        batch_size, num_nodes, features = x.size()
        
        # Create temporal embeddings
        if time_idx is None:
            time_idx = torch.arange(self.time_steps, device=x.device).float().view(-1, 1)
        time_embeddings = self.time_encoder(time_idx)  # [time_steps, hidden_dim]
        
        # Process each time step
        temporal_outputs = []
        for t in range(self.time_steps):
            # Get node features at time step t
            node_feats = x[:, :, t].reshape(-1, 1)  # [batch_size * num_nodes, 1]
            
            # Encode node features
            node_embeddings = self.feature_encoder(node_feats)  # [batch_size * num_nodes, hidden_dim]
            
            # Add temporal information
            time_embed = time_embeddings[t].expand(node_embeddings.size())
            node_time_embeddings = node_embeddings + time_embed
            
            # Apply first GAT layer
            h = self.gat1(node_time_embeddings, edge_index)
            h = self.relu(h)
            h = self.dropout(h)
            
            # Store output for this time step
            temporal_outputs.append(h)
        
        # Apply temporal attention
        temporal_weights = torch.nn.functional.softmax(self.temporal_attention, dim=0)  # [time_steps]
        
        # Weighted sum of temporal embeddings
        combined_h = torch.zeros_like(temporal_outputs[0])
        for t in range(self.time_steps):
            combined_h += temporal_outputs[t] * temporal_weights[t]
        
        # Apply final GAT layer
        out = self.gat2(combined_h, edge_index)
        
        return out.view(batch_size, num_nodes, -1)

# TGFormer model class definition
class TGFormer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout_rate=0.5):
        super(TGFormer, self).__init__()
        from torch_geometric.nn import TransformerConv
        
        # First linear layer to process features
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        
        # Transformer-based Graph Convolution
        self.transformer_conv1 = TransformerConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout_rate)
        self.transformer_conv2 = TransformerConv(hidden_dim*heads, output_dim, heads=1, dropout=dropout_rate, concat=False)
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
    
    def forward(self, x, edge_index):
        # Process input features first through linear layer
        x = self.linear1(x)
        x = self.relu(x)
        
        # First Transformer Conv layer
        x = self.transformer_conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output Transformer Conv layer
        x = self.transformer_conv2(x, edge_index)
        return x

def load_models():
    """Load all saved models"""
    logger.info("Loading models...")
    
    global models, node_mapping, edge_index
    
    try:
        # Load models if they exist
        if os.path.exists('saved_models'):
            # Model parameters - should match your training configuration
            input_features = 27 * 5  # 27 nodes × 5 time steps
            hidden_neurons = 8
            output_features = 27
            dropout_rate = 0.5
            
            # Initialize model classes with correct parameters
            mlp_model = MLP(input_features, hidden_neurons, output_features, dropout_rate)
            gcn_model = GCN(5, hidden_neurons, 1, dropout_rate)  # 5 = seq_length
            
            # IMPORTANT: Use hidden_neurons=4 for GAT as in your training code
            gat_model = GAT(5, 4, 1, heads=6, dropout_rate=dropout_rate)
            
            tgat_model = TGAT(1, hidden_neurons, 5, 1, heads=6, dropout_rate=dropout_rate)
            tgformer_model = TGFormer(5, 16, 1, heads=4, dropout_rate=dropout_rate)
            
            # Load model weights
            model_paths = {
                'MLP': 'saved_models/MLP.pt',
                'GCN': 'saved_models/GCN.pt',
                'GAT': 'saved_models/GAT.pt',
                'TGAT': 'saved_models/TGAT.pt',
                'TGFormer': 'saved_models/TGFormer.pt'
            }
            
            model_instances = {
                'MLP': mlp_model,
                'GCN': gcn_model,
                'GAT': gat_model,
                'TGAT': tgat_model,
                'TGFormer': tgformer_model
            }
            
            # Load weights for each model with weights_only=False
            for model_name, model in model_instances.items():
                path = model_paths[model_name]
                if os.path.exists(path):
                    # Use weights_only=False to handle older PyTorch versions
                    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
                    model.eval()
                    models[model_name] = model
                    logger.info(f"Loaded {model_name} from {path}")
                else:
                    logger.warning(f"Model file not found: {path}")
        
        # Load validation data with weights_only=False
        if os.path.exists('deployment_assets/validation_data.pt'):
            validation_data = torch.load('deployment_assets/validation_data.pt', map_location='cpu', weights_only=False)
            edge_index = validation_data.get('edge_index')
            node_mapping = validation_data.get('node_mapping', {})
            logger.info("Loaded validation data")
        elif os.path.exists('deployment_assets/sample_data.pt'):
            sample_data = torch.load('deployment_assets/sample_data.pt', map_location='cpu', weights_only=False)
            edge_index = sample_data.get('edge_index')
            node_mapping = sample_data.get('sample_node_mapping', {})
            logger.info("Loaded sample data")
        else:
            # Create a dummy edge_index if no data is found
            logger.warning("No validation data found. Creating dummy edge index.")
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        
        logger.info(f"Loaded {len(models)} models")
        return True
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions with the loaded models"""
    try:
        # Get data from request
        data = request.json
        
        # Extract sequence data and model choice
        sequence = data.get('sequence')
        model_name = data.get('model', 'TGFormer')  # Default to TGFormer if not specified
        
        if not sequence:
            return jsonify({'error': 'No sequence data provided'}), 400
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available', 
                           'available_models': list(models.keys())}), 400
        
        # Convert sequence to tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        
        # Make prediction based on model type
        if model_name == 'MLP':
            # For MLP, flatten the sequence
            if sequence_tensor.dim() == 1:
                # Already flattened
                flattened = sequence_tensor.unsqueeze(0)  # Add batch dimension
            else:
                # Needs flattening
                flattened = sequence_tensor.reshape(1, -1)  # Flatten to [1, n_features]
            prediction = models[model_name](flattened)
            
        elif model_name == 'TGAT':
            # For TGAT, special handling is needed
            logger.info(f"TGAT input original shape: {sequence_tensor.shape}")
            
            # For the case when input is [seq_length, num_nodes]
            if sequence_tensor.shape[0] == 5 and sequence_tensor.shape[1] == 27:
                # Transpose to [num_nodes, seq_length] first
                sequence_tensor = sequence_tensor.transpose(0, 1)
                logger.info(f"Transposed to: {sequence_tensor.shape}")
            
            # Create a batch of size 1 for each node
            batched_predictions = []
            
            # Process each node separately to avoid the error
            for node_idx in range(sequence_tensor.shape[0]):
                # Extract this node's features and add batch and node dimensions
                node_features = sequence_tensor[node_idx].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_length]
                
                # Make sure it's the right shape for the model
                dummy_edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long).t()
                
                try:
                    # Predict for this single node
                    node_pred = models[model_name](node_features, dummy_edge_index)
                    batched_predictions.append(node_pred.squeeze().detach().cpu().item())
                except Exception as e:
                    logger.error(f"TGAT node prediction error: {str(e)}")
                    return jsonify({'error': f'TGAT prediction failed: {str(e)}'}), 500
            
            # Combine all node predictions
            prediction = torch.tensor([batched_predictions])
            logger.info(f"TGAT combined prediction shape: {prediction.shape}")
            
        else:
            # For other graph models (GCN, GAT, TGFormer)
            prediction = models[model_name](sequence_tensor, edge_index)
        
        # Convert prediction to list
        prediction_list = prediction.detach().cpu().numpy().tolist()
        
        # If we have node mapping, include node names in response
        if node_mapping and isinstance(prediction_list[0], list):
            reverse_mapping = {idx: node for node, idx in node_mapping.items()}
            labeled_predictions = {reverse_mapping.get(i, f"node_{i}"): float(val) 
                                  for i, val in enumerate(prediction_list[0])}
            
            return jsonify({
                'model': model_name,
                'predictions': prediction_list,
                'labeled_predictions': labeled_predictions
            })
        
        return jsonify({
            'model': model_name,
            'predictions': prediction_list
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Return information about available models"""
    return jsonify({
        'available_models': list(models.keys()),
        'node_count': len(node_mapping) if node_mapping else 'unknown',
        'edge_count': edge_index.shape[1] // 2 if edge_index is not None else 'unknown'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    if models and edge_index is not None:
        return jsonify({'status': 'healthy', 'models_loaded': len(models)})
    else:
        return jsonify({'status': 'unhealthy', 'reason': 'Models not loaded'}), 503

if __name__ == '__main__':
    # Load models on startup
    success = load_models()
    
    if success:
        logger.info("Models loaded successfully, starting API server")
        # Get port from environment or use default
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
    else:
        logger.error("Failed to load models, exiting")
