import requests
import json
import numpy as np
import random
import time

# Azure deployment FQDN
base_url = "<insert base url>"

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("Generating synthetic test data...")
# Parameters
num_nodes = 27
seq_length = 5

# Create synthetic data for each model
# 1. MLP - Flattened data with shape [num_nodes * seq_length]
mlp_data = np.random.uniform(-1.0, 1.0, num_nodes * seq_length).tolist()

# 2. GCN/GAT/TGFormer - Node-centric data with shape [num_nodes, seq_length]
graph_data = []
for i in range(num_nodes):
    # Create a time series for each node
    node_trend = np.linspace(-1, 1, seq_length)
    node_noise = np.random.normal(0, 0.2, seq_length)
    node_data = (node_trend + node_noise).tolist()
    graph_data.append(node_data)

# 3. TGAT - Time-centric data with shape [seq_length, num_nodes]
tgat_data = np.array(graph_data).T.tolist()  # Transpose the graph data

# Test health endpoint
print("\nTesting health endpoint...")
response = requests.get(f"{base_url}/health")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error: {response.text}")

# Test models endpoint
print("\nTesting models endpoint...")
response = requests.get(f"{base_url}/models")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error: {response.text}")

# Define test cases with appropriate data for each model
model_configs = {
    "MLP": {
        "description": "Multilayer Perceptron - Flattened input with 135 values",
        "input": mlp_data
    },
    "GCN": {
        "description": "Graph Convolutional Network - Node features across time",
        "input": graph_data
    },
    "GAT": {
        "description": "Graph Attention Network - Node features with attention",
        "input": graph_data
    },
    "TGAT": {
        "description": "Temporal Graph Attention Network - Time steps for all nodes",
        "input": tgat_data
    },
    "TGFormer": {
        "description": "Transformer Graph Network - Node features with transformer architecture",
        "input": graph_data
    }
}

# Test prediction for each model
for model_name, config in model_configs.items():
    print(f"\nTesting {model_name}: {config['description']}")
    print("-" * 80)
    
    # Prepare the request data
    request_data = {
        "model": model_name,
        "sequence": config["input"]
    }
    
    # Make the API request
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json=request_data,
            timeout=30  # Increased timeout for larger models
        )
        elapsed_time = time.time() - start_time
        
        print(f"Status: {response.status_code}, Response Time: {elapsed_time:.3f} seconds")
        
        if response.status_code == 200:
            prediction = response.json()
            
            # Print prediction info
            if 'predictions' in prediction:
                pred_data = prediction['predictions']
                
                if isinstance(pred_data, list) and len(pred_data) > 0:
                    if isinstance(pred_data[0], list):
                        print(f"Prediction Shape: {len(pred_data)} x {len(pred_data[0])}")
                        print(f"Sample Predictions (first node, first 5 values):")
                        print(f"  {pred_data[0][:5]}")
                    else:
                        print(f"Prediction Length: {len(pred_data)}")
                        print(f"Sample Predictions: {pred_data[:5]}")
                else:
                    print(f"Prediction Data: {pred_data}")
                    
                # Show labeled predictions if available
                if 'labeled_predictions' in prediction:
                    labeled = prediction['labeled_predictions']
                    print("\nLabeled Predictions (sample):")
                    count = 0
                    for key, value in labeled.items():
                        if count < 5:
                            print(f"  {key}: {value}")
                            count += 1
                        else:
                            print("  ...")
                            break
            else:
                print(f"Response: {prediction}")
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Request failed: {str(e)}")

print("\nAPI testing completed.")
