# deploy_to_azure.py
import os
import argparse
import subprocess
import json
import time
import random
import string

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Deploy GNN model to Azure')
    parser.add_argument('--resource-group', type=str, required=True,
                        help='Azure resource group name')
    parser.add_argument('--location', type=str, default='eastus',
                        help='Azure region (default: eastus)')
    parser.add_argument('--container-name', type=str, default='gnn-prediction-service',
                        help='Name for the container instance')
    parser.add_argument('--registry-name', type=str,
                        help='Azure Container Registry name. If not specified, a new one will be created.')
    parser.add_argument('--deployment-type', type=str, choices=['aci', 'function'], default='aci',
                        help='Deployment type: Azure Container Instance (aci) or Azure Function (function)')
    parser.add_argument('--cpu', type=float, default=1.0,
                        help='CPU cores for container (default: 1.0)')
    parser.add_argument('--memory', type=float, default=1.5,
                        help='Memory in GB for container (default: 1.5)')
    return parser.parse_args()

def run_command(cmd, description=None):
    """Run a shell command and return the output"""
    if description:
        print(f"\n>>> {description}...")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    return result.stdout.strip()

def create_resource_group(args):
    """Create Azure resource group if it doesn't exist"""
    # Check if resource group exists
    cmd = ['az', 'group', 'exists', '--name', args.resource_group]
    exists = run_command(cmd, "Checking if resource group exists")
    
    if exists == 'false':
        cmd = ['az', 'group', 'create', 
               '--name', args.resource_group, 
               '--location', args.location]
        run_command(cmd, f"Creating resource group '{args.resource_group}'")
        return True
    
    return exists == 'true'

def create_container_registry(args):
    """Create Azure Container Registry if needed"""
    registry_name = args.registry_name
    
    if not registry_name:
        # Generate a unique name if not provided
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        registry_name = f"gnnregistry{suffix}"
    
    # Check if registry exists
    cmd = ['az', 'acr', 'show', '--name', registry_name, '--resource-group', args.resource_group]
    registry_exists = run_command(cmd)
    
    if not registry_exists:
        cmd = ['az', 'acr', 'create', 
               '--resource-group', args.resource_group, 
               '--name', registry_name, 
               '--sku', 'Basic', 
               '--admin-enabled', 'true']
        run_command(cmd, f"Creating container registry '{registry_name}'")
    
    # Get registry credentials
    cmd = ['az', 'acr', 'credential', 'show', 
           '--name', registry_name, 
           '--resource-group', args.resource_group]
    credentials = json.loads(run_command(cmd, "Getting registry credentials"))
    
    return {
        'name': registry_name,
        'loginServer': f"{registry_name}.azurecr.io",
        'username': credentials['username'],
        'password': credentials['passwords'][0]['value']
    }

def build_and_push_container(registry_info):
    """Build and push the Docker container to ACR"""
    # Login to registry
    cmd = ['az', 'acr', 'login', '--name', registry_info['name']]
    run_command(cmd, "Logging into container registry")
    
    # Build the container image
    image_name = f"{registry_info['loginServer']}/gnn-prediction:latest"
    
    # Check if we need to copy the Dockerfile
    if not os.path.exists('Dockerfile'):
        print("No Dockerfile found in current directory. Using existing Dockerfile.")
    
    # Build the container
    print("Building container image...")
    cmd = ['docker', 'build', '-t', image_name, '.']
    build_result = run_command(cmd, "Building container image")
    
    if not build_result:
        print("Warning: Docker build command may have failed. Checking if image exists...")
        cmd = ['docker', 'images', image_name, '--format', '{{.Repository}}']
        image_exists = run_command(cmd)
        if not image_exists:
            print("Error: Failed to build image and no existing image found.")
            return None
    
    # Push to registry
    cmd = ['docker', 'push', image_name]
    run_command(cmd, "Pushing container image to registry")
    
    return image_name

def save_deployment_info(args, registry_info, fqdn):
    """Save deployment information to a file"""
    deployment_info = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'resource_group': args.resource_group,
        'location': args.location,
        'container_name': args.container_name,
        'registry_name': registry_info['name'],
        'registry_server': registry_info['loginServer'],
        'endpoint': f"http://{fqdn}:5000",
        'health_check': f"http://{fqdn}:5000/health",
        'models_info': f"http://{fqdn}:5000/models",
        'prediction_endpoint': f"http://{fqdn}:5000/predict"
    }
    
    # Save to file
    with open('deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"Deployment information saved to deployment_info.json")

def deploy_to_aci(args, registry_info, image_name):
    """Deploy the container to Azure Container Instances"""
    container_name = args.container_name
    
    # Create container instance
    cmd = ['az', 'container', 'create',
           '--resource-group', args.resource_group,
           '--name', container_name,
           '--image', image_name,
           '--cpu', str(args.cpu),
           '--memory', str(args.memory),
           '--registry-login-server', registry_info['loginServer'],
           '--registry-username', registry_info['username'],
           '--registry-password', registry_info['password'],
           '--dns-name-label', container_name.lower(),
           '--ports', '5000',
           '--environment-variables', 'PORT=5000',
           '--tags', 'purpose=prediction-service,project=gnn-models,deployment-date=' + time.strftime('%Y-%m-%d')]
    
    deployment_result = run_command(cmd, "Deploying to Azure Container Instance")
    
    if deployment_result:
        # Get container info
        cmd = ['az', 'container', 'show',
               '--resource-group', args.resource_group,
               '--name', container_name,
               '--query', 'ipAddress.fqdn',
               '--output', 'tsv']
        fqdn = run_command(cmd, "Getting container FQDN")
        
        if fqdn:
            print(f"\nDeployment successful!")
            print(f"API Endpoint: http://{fqdn}:5000")
            print(f"Health Check: http://{fqdn}:5000/health")
            print(f"Models Info: http://{fqdn}:5000/models")
            print(f"Make Predictions: POST http://{fqdn}:5000/predict")
            
            # Save deployment information
            save_deployment_info(args, registry_info, fqdn)
            
            return True
    
    return False

def deploy_to_function(args):
    """Deploy as an Azure Function"""
    # This is a simplified version - in a real scenario you would need more setup
    function_name = args.container_name
    
    # Create storage account for function
    storage_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    storage_name = f"gnnfunctions{storage_suffix}"
    
    cmd = ['az', 'storage', 'account', 'create',
           '--name', storage_name,
           '--resource-group', args.resource_group,
           '--location', args.location,
           '--sku', 'Standard_LRS']
    run_command(cmd, "Creating storage account for Function App")
    
    # Create function app
    cmd = ['az', 'functionapp', 'create',
           '--name', function_name,
           '--resource-group', args.resource_group,
           '--storage-account', storage_name,
           '--consumption-plan-location', args.location,
           '--runtime', 'python',
           '--runtime-version', '3.9',
           '--functions-version', '4']
    deployment_result = run_command(cmd, "Creating Function App")
    
    if deployment_result:
        print(f"\nFunction App created: {function_name}")
        print("You'll need to deploy your code to the Function App.")
        print("See https://docs.microsoft.com/en-us/azure/azure-functions/functions-deployment-technologies")
        return True
    
    return False

def main():
    """Main deployment function"""
    args = parse_args()
    
    print("Starting deployment to Azure...")
    
    # Create resource group if needed
    if not create_resource_group(args):
        print("Failed to create resource group. Aborting.")
        return
    
    if args.deployment_type == 'aci':
        # Create container registry
        registry_info = create_container_registry(args)
        if not registry_info:
            print("Failed to create or access container registry. Aborting.")
            return
        
        # Build and push container
        image_name = build_and_push_container(registry_info)
        if not image_name:
            print("Failed to build and push container. Aborting.")
            return
        
        # Deploy to ACI
        if not deploy_to_aci(args, registry_info, image_name):
            print("Failed to deploy to Azure Container Instance. Aborting.")
            return
    
    elif args.deployment_type == 'function':
        # Deploy as Azure Function
        if not deploy_to_function(args):
            print("Failed to deploy Azure Function. Aborting.")
            return
    
    print("\nDeployment completed successfully!")

if __name__ == "__main__":
    main()
