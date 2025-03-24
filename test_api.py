import requests
import json
import torch
import numpy as np

# Load the validation data
try:
    print("Loading validation data...")
    validation_data = torch.load('deployment_assets/validation_data.pt', weights_only=False)
    
    # Extract test data based on model type
    if 'X_test_tensor' in validation_data:
        # For MLP - use the first sample
        mlp_input = validation_data['X_test_tensor'][0].numpy()
        mlp_input_shape = mlp_input.shape
        
        # Reshape for API request
        mlp_sequence = mlp_input.reshape(-1).tolist()
        
        print(f"MLP Input Shape: {mlp_input_shape}")
        print(f"MLP Input Sample: {mlp_sequence[:10]}...")  # Show first 10 values
    else:
        print("Warning: No MLP test data found.")
        mlp_sequence = [0.1] * (27 * 5)  # Default test data
    
    # For graph models
    if 'test_data_list' in validation_data and len(validation_data['test_data_list']) > 0:
        # Get the first test item
        test_item = validation_data['test_data_list'][0]
        
        # Extract features (x) and reshape appropriately
        graph_input = test_item.x.numpy()
        graph_input_shape = graph_input.shape
        
        # For GCN, GAT, TGFormer - input shape should be [num_nodes, seq_length]
        # Convert the numpy array to list for JSON serialization
        graph_sequence = graph_input.tolist()
        
        # For TGAT - needs to be transposed to [seq_length, num_nodes]
        tgat_sequence = np.transpose(graph_input).tolist()
        
        print(f"Graph Model Input Shape: {graph_input_shape}")
        print(f"Graph Input Sample (first node): {graph_sequence[0]}")
    else:
        print("Warning: No graph model test data found.")
        # Default test data for graph models
        graph_sequence = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(27)]  # [num_nodes, seq_length]
        tgat_sequence = [[0.1] * 27, [0.2] * 27, [0.3] * 27, [0.4] * 27, [0.5] * 27]  # [seq_length, num_nodes]
    
    print("\nTest data prepared successfully.")

except Exception as e:
    print(f"Error loading validation data: {str(e)}")
    # Create default test data
    mlp_sequence = [0.1] * (27 * 5)
    graph_sequence = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(27)]
    tgat_sequence = [[0.1] * 27, [0.2] * 27, [0.3] * 27, [0.4] * 27, [0.5] * 27]
    print("Using default test data.")

print("\n" + "="*50)

# 1. Test health endpoint
print("\nTesting health endpoint...")
response = requests.get("http://127.0.0.1:5000/health")
print(f"Status: {response.status_code}")
print(response.json())

print("\n" + "="*50)

# 2. Test models endpoint
print("\nTesting models endpoint...")
response = requests.get("http://127.0.0.1:5000/models")
print(f"Status: {response.status_code}")
print(response.json())

print("\n" + "="*50)

# 3. Test prediction with each model
model_configs = {
    "MLP": {"input": mlp_sequence, "reshaper": lambda x: x},  # MLP uses flattened input
    "GCN": {"input": graph_sequence, "reshaper": lambda x: x},  # GCN uses [num_nodes, seq_length]
    "GAT": {"input": graph_sequence, "reshaper": lambda x: x},  # GAT uses [num_nodes, seq_length]
    "TGAT": {"input": tgat_sequence, "reshaper": lambda x: x},  # TGAT uses [seq_length, num_nodes]
    "TGFormer": {"input": graph_sequence, "reshaper": lambda x: x}  # TGFormer uses [num_nodes, seq_length]
}

for model_name, config in model_configs.items():
    print(f"\nTesting prediction with {model_name}...")
    
    # Get appropriate input for this model
    model_input = config["input"]
    
    # Print input information
    print(f"Input type: {type(model_input)}")
    if isinstance(model_input[0], list):
        print(f"Input shape: {len(model_input)} x {len(model_input[0])}")
    else:
        print(f"Input length: {len(model_input)}")
    
    # Prepare request data
    data = {
        "model": model_name,
        "sequence": model_input
    }
    
    # Send POST request
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json=data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        prediction = response.json()
        
        # Print prediction info
        if 'predictions' in prediction:
            pred_data = prediction['predictions']
            print(f"Prediction data type: {type(pred_data)}")
            
            if isinstance(pred_data, list) and len(pred_data) > 0:
                if isinstance(pred_data[0], list):
                    print(f"Prediction shape: {len(pred_data)} x {len(pred_data[0])}")
                else:
                    print(f"Prediction length: {len(pred_data)}")
                
                # Print full prediction for small outputs, or a sample for large ones
                if len(str(pred_data)) < 1000:
                    print(f"Full prediction: {pred_data}")
                else:
                    if isinstance(pred_data[0], list):
                        sample = pred_data[0][:5]  # First 5 elements of first row
                    else:
                        sample = pred_data[:5]  # First 5 elements
                    print(f"Prediction sample: {sample}...")
            else:
                print(f"Prediction: {pred_data}")
        else:
            print(f"Response: {prediction}")
    else:
        print(f"Error: {response.text}")

print("\n" + "="*50)
print("\nAPI testing completed.")
