#!/bin/bash
set -e

echo "Deploying Inference Demo Infrastructure..."

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

# Generate terraform.tfvars with all values from .env
cat > terraform.tfvars << EOF
# VCD Configuration
vcd_user                = "${VCD_USER}"
vcd_password            = "${VCD_PASSWORD}"
vcd_org                 = "${VCD_ORG}"
vcd_vdc                 = "${VCD_VDC}"
vcd_url                 = "${VCD_URL}"

# Network Configuration
network_segment         = "${NETWORK_SEGMENT}"
default_gateway        = "${DEFAULT_GATEWAY}"
dns_servers           = ${DNS_SERVERS}
vm_ip                 = "${VM_IP}"

# VM Configuration
vm_name               = "${VM_NAME}"
template_name         = "${TEMPLATE_NAME}"
vm_catalog_name       = "${VM_CATALOG_NAME}"

# Hardware Configuration
vm_memory               = ${VM_MEMORY}
vm_cpus                 = ${VM_CPUS}
vm_cpu_cores            = ${VM_CPU_CORES}
vm_cpu_hot_add_enabled  = ${VM_CPU_HOT_ADD_ENABLED}
vm_memory_hot_add_enabled = ${VM_MEMORY_HOT_ADD_ENABLED}
vm_disk_size            = ${VM_DISK_SIZE}

# SSH Configuration
ssh_public_key          = "${SSH_PUBLIC_KEY}"
EOF

# Initialize and apply Terraform
terraform init
terraform plan
terraform apply -auto-approve

# Get the VM IP
VM_IP=$(terraform output -raw vm_ip)

echo "Deployment complete! Your inference demo is being set up at: http://${VM_IP}"
echo "Please allow 5-10 minutes for the initial setup to complete."
echo "You can monitor the setup progress using: ssh ubuntu@${VM_IP} 'tail -f /var/log/cloud-init-output.log'"
echo "To check the application logs: ssh ubuntu@${VM_IP} 'tail -f /var/log/inference-demo.out.log'"