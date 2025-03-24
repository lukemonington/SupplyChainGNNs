import requests
import json
import numpy as np
import random
import time

# API endpoint
base_url = "http://127.0.0.1:5000"

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
print(f"\nMLP Input:")
print(f"- Shape: [{len(mlp_data)}]")
print(f"- Example (first 10 values): {mlp_data[:10]}")
print(f"- Description: Flattened array with {len(mlp_data)} values representing 27 nodes x 5 time steps")

# 2. GCN/GAT/TGFormer - Node-centric data with shape [num_nodes, seq_length]
graph_data = []
for i in range(num_nodes):
    # Create a time series for each node
    # Generate a trend sequence with some randomness
    node_trend = np.linspace(-1, 1, seq_length)
    node_noise = np.random.normal(0, 0.2, seq_length)
    node_data = (node_trend + node_noise).tolist()
    graph_data.append(node_data)

print(f"\nGCN/GAT/TGFormer Input:")
print(f"- Shape: [{len(graph_data)} x {len(graph_data[0])}]")
print(f"- Example (first 3 nodes):")
for i in range(min(3, len(graph_data))):
    print(f"  Node {i}: {graph_data[i]}")
print(f"- Description: Each node has a sequence of 5 values representing time steps")

# 3. TGAT - Time-centric data with shape [seq_length, num_nodes]
tgat_data = np.array(graph_data).T.tolist()  # Transpose the graph data
print(f"\nTGAT Input:")
print(f"- Shape: [{len(tgat_data)} x {len(tgat_data[0])}]")
print(f"- Example (first 2 time steps):")
for i in range(min(2, len(tgat_data))):
    print(f"  Time Step {i}: First 5 nodes: {tgat_data[i][:5]}")
print(f"- Description: Each time step has values for all 27 nodes")

print("\n" + "="*50)

# Test health endpoint
print("\nTesting health endpoint...")
response = requests.get(f"{base_url}/health")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.text}")

print("\n" + "="*50)

# Test models endpoint
print("\nTesting models endpoint...")
response = requests.get(f"{base_url}/models")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.text}")

print("\n" + "="*50)

# Define test cases with synthetic data
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
    
    # Prepare sample data to show in request
    data = config["input"]
    
    # Show example of the request data
    print("Example Request:")
    request_data = {
        "model": model_name,
        "sequence": data  # In a real example we'd truncate this for display
    }
    
    # Print truncated/simplified JSON for display
    if isinstance(data[0], list):
        display_data = [data[0]]
        if len(data) > 1:
            display_data.append(data[1])
        if len(data) > 2:
            display_data.append(["..."])
    else:
        display_data = data[:5] + ["..."] + data[-5:] if len(data) > 10 else data
    
    display_request = {
        "model": model_name,
        "sequence": display_data
    }
    print(json.dumps(display_request, indent=2))
    
    # Make the actual API request
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json=request_data
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
                        print(f"Sample Predictions (first 3 items):")
                        for i, row in enumerate(pred_data[:3]):
                            if len(row) > 5:
                                print(f"  Item {i}: {row[:5]} ... (truncated)")
                            else:
                                print(f"  Item {i}: {row}")
                    else:
                        print(f"Prediction Length: {len(pred_data)}")
                        if len(pred_data) > 10:
                            print(f"Sample Predictions: {pred_data[:5]} ... (truncated)")
                        else:
                            print(f"Full Predictions: {pred_data}")
                else:
                    print(f"Prediction Data: {pred_data}")
                    
                # If there are labeled predictions, show a sample
                if 'labeled_predictions' in prediction:
                    labeled = prediction['labeled_predictions']
                    print("\nLabeled Predictions (sample):")
                    shown = 0
                    for key, value in labeled.items():
                        if shown < 5:  # Show just a few examples
                            print(f"  {key}: {value}")
                            shown += 1
                        else:
                            print("  ...")
                            break
            else:
                print(f"Response: {prediction}")
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Request failed: {str(e)}")

print("\n" + "="*50)
print("\nAPI testing with synthetic data completed.")
