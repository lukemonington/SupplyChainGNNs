# Supply Chain Forecasting with Graph Neural Networks

## Project Overview
This repository implements and compares various Graph Neural Network (GNN) architectures for supply chain demand forecasting, with a focus on leveraging network relationships between products to improve prediction accuracy. The project demonstrates how GNNs can capture complex interdependencies in supply chains that traditional forecasting methods miss.

### Key Features
- Implementation of 5 neural network architectures:
  - Multi-Layer Perceptron (MLP) as baseline
  - Graph Convolutional Network (GCN)
  - Graph Attention Network (GAT)
  - Temporal Graph Attention Network (TGAT)
  - Transformer-based Graph Neural Network (TGFormer)
- Comprehensive model evaluation and comparison
- Production-ready API for model inference
- Azure deployment solution with Docker containerization

## Dataset
This project utilizes the SupplyGraph benchmark dataset (2024), which contains real-world supply chain data from a FMCG company in Bangladesh. Key dataset characteristics:
- 40 distinct product nodes with various relationships forming a graph structure
- Temporal features spanning January-August 2023
- Four key metrics: production output, sales orders, deliveries, and factory issues
- Each product has connections to others through shared facilities or product categories

## Model Architectures

### MLP (Baseline)
- Standard feed-forward neural network
- Input: Flattened sequences of all 27 products across 5 time steps
- Hidden layer: 8 neurons with ReLU activation and dropout
- Output: Predictions for all products simultaneously

### GCN (Graph Convolutional Network)
- Leverages graph structure of supply chain
- Each node processes its own time series while aggregating information from connected products
- Two GCN layers with dropout and ReLU activation
- Captures shared manufacturing constraints and product group relationships

### GAT (Graph Attention Network)
- Extends GCN with attention mechanism
- 6 attention heads to dynamically weight neighbor importance
- Allows model to identify which product relationships are most predictive

### TGAT (Temporal Graph Attention Network)
- Specialized for temporal dynamics in graphs
- Separate encoders for time steps and node features
- Dual attention mechanism for both spatial connections and temporal patterns
- Learned temporal attention combines information across time steps

### TGFormer (Transformer-based Graph Neural Network)
- Cutting-edge architecture adapting transformer self-attention to graphs
- TransformerConv layers with multi-head attention (4 heads)
- Linear projection of time series features before graph processing
- Computes attention scores between connected nodes in supply chain

## Training Process
The training pipeline includes:
1. Data preprocessing with rolling time windows (sequence length 5)
2. Z-score normalization of temporal features
3. 95:5 train-test split
4. Adam optimizer with learning rate 0.001
5. Mean Squared Error (MSE) loss function
6. Training for 200 epochs with early stopping (patience 20)
7. Evaluation using RMSE, MAE, and correlation metrics

## Deployment on Azure

### Prerequisites
- Azure account with subscription
- Azure CLI installed and configured
- Docker installed locally

### Deployment Steps

1. **Build and push Docker image to Azure Container Registry (ACR)**
```bash
# Create ACR if it doesn't exist
az acr create --resource-group myResourceGroup --name myACRRegistry --sku Basic

# Build and push image
az acr build --registry myACRRegistry --image supply-chain-gnn:latest .
```

2. **Deploy to Azure Container Instances (for testing)**
```bash
az container create \\
  --resource-group myResourceGroup \\
  --name supply-chain-gnn-api \\
  --image myACRRegistry.azurecr.io/supply-chain-gnn:latest \\
  --dns-name-label supply-chain-gnn \\
  --ports 5000
```

3. **Deploy to Azure Kubernetes Service (AKS) for production**
```bash
# Create AKS cluster if it doesn't exist
az aks create \\
  --resource-group myResourceGroup \\
  --name myAKSCluster \\
  --node-count 2 \\
  --enable-addons monitoring \\
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group myResourceGroup --name myAKSCluster

# Deploy using Kubernetes manifests
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/hpa.yaml
```

### Azure DevOps CI/CD Pipeline
The repository includes Azure DevOps pipeline definitions for:
- Automated testing on pull requests
- Model validation against benchmarks
- Docker image building and pushing to ACR
- Deployment to staging and production environments with approvals

## API Usage
The deployed API exposes the following endpoints:

### Health Check
```
GET /health
```
Returns status of the API and number of loaded models.

### List Models
```
GET /models
```
Returns information about available models, node count, and edge count.

### Make Predictions
```
POST /predict
```
Request body:
```json
{
  "model": "GCN",  // Options: MLP, GCN, GAT, TGAT, TGFormer
  "sequence": [...]  // Input sequence in appropriate format
}
```

Input format depends on the model:
- MLP: Flattened array [num_nodes * seq_length]
- GCN/GAT/TGFormer: Node-centric data [num_nodes, seq_length]
- TGAT: Time-centric data [seq_length, num_nodes]

Response:
```json
{
  "model": "GCN",
  "predictions": [...],
  "labeled_predictions": {
    "node_0": 0.75,
    "node_1": 0.62,
    ...
  }
}
```

## Results and Key Findings

| Model     | Training RMSE | Testing RMSE | Testing MAE | Parameters |
|-----------|---------------|--------------|-------------|------------|
| MLP       | 0.7103        | 1.0657       | 0.7843      | 1,208      |
| GCN       | 0.8163        | 0.8005       | 0.6151      | 464        |
| GAT       | 0.8275        | 0.8302       | 0.6378      | 626        |
| TGAT      | 0.8492        | 0.8210       | 0.6343      | 798        |
| TGFormer  | 0.7843        | 0.8098       | 0.6122      | 734        |

Key findings:
1. MLP shows classic signs of overfitting—lowest training error but highest testing error
2. GCN demonstrates exceptional generalization with test loss lower than training loss
3. TGFormer achieves the best raw test metrics but shows higher propensity for overfitting
4. Graph-based models consistently outperform the MLP baseline, validating the value of graph structure
5. For production environments, GCN provides the best balance of performance and reliability

## Dependencies
Major dependencies include:
- PyTorch 2.6.0
- PyTorch Geometric 2.6.1
- Flask 3.1.0
- Pandas 2.2.2
- Scikit-learn 1.6.1
- Matplotlib 3.10.0

See `requirements.txt` for a complete list.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Resources
[1] Paper introducing the dataset: https://arxiv.org/pdf/2401.15299 
[2] Research paper working with the SupplyGraph dataset that I implemented: https://arxiv.org/html/2408.14501v1#:~:text=The%20exploratory%20data%20analysis%20on,the%20demand%20forecasting%20use%20case
[3] Paper introducing the TGFormer: https://dtclee1222.github.io/_pages/paper/2025-TGformer.pdf 
