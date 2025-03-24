# Graph Neural Network Models API for Demand Prediction

This API provides access to five different neural network models for demand prediction, each with different architectures and capabilities. The models range from simple multilayer perceptrons to sophisticated graph neural networks that leverage network relationships between products and suppliers.

## Available Models

The API hosts five models with different architectures:

| Model | Type | Description |
|-------|------|-------------|
| MLP | Multilayer Perceptron | A traditional feedforward neural network |
| GCN | Graph Convolutional Network | Uses graph convolutions to leverage network structure |
| GAT | Graph Attention Network | Employs attention mechanisms to weight neighbor importance |
| TGAT | Temporal Graph Attention Network | Adds temporal awareness to graph attention |
| TGFormer | Transformer Graph Network | Leverages transformer architecture with graph structure |

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the API and the number of loaded models.

**Example Response:**
```json
{
  "status": "healthy",
  "models_loaded": 5
}
```

### Available Models
```
GET /models
```
Returns information about available models, number of nodes, and edges in the graph.

**Example Response:**
```json
{
  "available_models": ["MLP", "GCN", "GAT", "TGAT", "TGFormer"],
  "node_count": 27,
  "edge_count": 1343
}
```

### Make Predictions
```
POST /predict
```
Makes predictions using the specified model.

**Request Parameters:**
- `model`: Name of the model to use (one of "MLP", "GCN", "GAT", "TGAT", "TGFormer")
- `sequence`: Input data in the appropriate format for the model

**Example Request for MLP:**
```json
{
  "model": "MLP",
  "sequence": [0.1, 0.2, 0.3, ..., 0.8]  // Flattened input, length 135 (27 nodes * 5 time steps)
}
```

**Example Request for Graph Models (GCN, GAT, TGFormer):**
```json
{
  "model": "GCN",
  "sequence": [
    [0.1, 0.2, 0.3, 0.4, 0.5],  // Node 1 features across 5 time steps
    [0.2, 0.3, 0.4, 0.5, 0.6],  // Node 2 features
    // ... (27 nodes total)
  ]
}
```

**Example Request for TGAT:**
```json
{
  "model": "TGAT",
  "sequence": [
    [0.1, 0.2, 0.3, ...],  // Time step 1 features for all 27 nodes
    [0.2, 0.3, 0.4, ...],  // Time step 2 features for all 27 nodes
    // ... (5 time steps total)
  ]
}
```

**Example Response:**
```json
{
  "model": "GCN",
  "predictions": [
    [0.62], [0.69], [0.69], [0.69], [0.68], [0.69], [0.50], [0.46], [0.63], 
    [0.68], [0.68], [0.65], [0.56], [0.66], [0.73], [0.68], [0.35], [0.15], 
    [0.09], [0.15], [0.01], [0.16], [0.07], [0.03], [0.19], [0.12], [-0.06]
  ],
  "labeled_predictions": {
    "node_1": 0.62,
    "node_2": 0.69,
    // ...
  }
}
```

## Model Details

### MLP (Multilayer Perceptron)
- **Architecture**: 2-layer feedforward neural network with ReLU activation and dropout
- **Input**: Flattened sequence with shape `[batch_size, 135]` (27 nodes × 5 time steps)
- **Output**: Predictions for all 27 nodes with shape `[batch_size, 27]`
- **Parameters**:
  - Input features: 135 (27 nodes × 5 time steps)
  - Hidden neurons: 8
  - Output features: 27
  - Dropout rate: 0.5

### GCN (Graph Convolutional Network)
- **Architecture**: 2-layer graph convolutional network that leverages graph structure
- **Input**: Node features with shape `[27, 5]` (27 nodes, each with 5 time steps of features)
- **Output**: Predictions for all 27 nodes with shape `[27, 1]`
- **Parameters**:
  - Input features: 5 (sequence length)
  - Hidden neurons: 8
  - Output features: 1
  - Dropout rate: 0.5

### GAT (Graph Attention Network)
- **Architecture**: Multi-head graph attention network with 6 attention heads
- **Input**: Node features with shape `[27, 5]` (27 nodes, each with 5 time steps of features)
- **Output**: Predictions for all 27 nodes with shape `[27, 1]`
- **Parameters**:
  - Input features: 5 (sequence length)
  - Hidden neurons: 4 (per attention head)
  - Attention heads: 6
  - Output features: 1
  - Dropout rate: 0.5

### TGAT (Temporal Graph Attention Network)
- **Architecture**: Graph attention network with temporal awareness
- **Input**: Can accept either:
  - Node-first format: `[27, 5]` (27 nodes, each with 5 time steps)
  - Time-first format: `[5, 27]` (5 time steps, each with 27 node features)
- **Output**: Predictions for all 27 nodes with shape `[1, 27]`
- **Parameters**:
  - Input features: 1 (single feature per node per time step)
  - Hidden neurons: 8
  - Time steps: 5
  - Attention heads: 6
  - Output features: 1
  - Dropout rate: 0.5

### TGFormer (Transformer Graph Network)
- **Architecture**: Graph network using transformer architecture with multi-head attention
- **Input**: Node features with shape `[27, 5]` (27 nodes, each with 5 time steps of features)
- **Output**: Predictions for all 27 nodes with shape `[27, 1]`
- **Parameters**:
  - Input features: 5 (sequence length)
  - Hidden neurons: 16
  - Attention heads: 4
  - Output features: 1
  - Dropout rate: 0.5

## Usage Examples

### Python Example
```python
import requests
import numpy as np

# Server URL
base_url = "http://your-deployment-url.com:5000"

# Example: Using GCN model
# Create input sequence with shape [27, 5]
sequence = np.random.random((27, 5)).tolist()

# Prepare request
data = {
    "model": "GCN",
    "sequence": sequence
}

# Make prediction
response = requests.post(f"{base_url}/predict", json=data)

# Print results
if response.status_code == 200:
    result = response.json()
    print(f"Model: {result['model']}")
    print(f"Predictions shape: {len(result['predictions'])}x{len(result['predictions'][0])}")
    print(f"First few predictions: {result['predictions'][:3]}")
else:
    print(f"Error: {response.text}")
```

### cURL Example
```bash
# Health check
curl http://your-deployment-url.com:5000/health

# Get available models
curl http://your-deployment-url.com:5000/models

# Make prediction with MLP (replace with actual input data)
curl -X POST http://your-deployment-url.com:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"model":"MLP","sequence":[0.1, 0.2, 0.3, ..., 0.5]}'
```


## Graph Structure

The models (except MLP) leverage a graph structure representing relationships between 27 nodes. The graph contains:
- 27 nodes representing different products/suppliers
- 1343 edges representing relationships between nodes

## Deployment

This API is deployed on Azure Container Instances, providing a scalable and cost-effective solution. The containerized approach allows for easy deployment and scaling based on demand.

## Requirements

- Python 3.11.11
- PyTorch 2.6.0
- PyTorch Geometric 2.6.1
- Flask 3.1.0
