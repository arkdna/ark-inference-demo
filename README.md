# ark Inference Demo

A CPU-optimized inference API deployment using Terraform and VMware Cloud Director.

## Features
- Flask-based API server
- Multiple inference model support with CPU optimizations
- Nginx reverse proxy

## Prerequisites
- Terraform >= 0.13
- VMware Cloud Director access
- Python 3.8+
- Git

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/arkdna/ark-inference-demo.git
   cd ark-inference-demo
   ```

2. Create and configure your environment file:
   ```bash
   cp .env.example .env
   ```

3. Edit .env and set your values:
   ```bash
   # Required values:
   VCD_PASSWORD=your-vcd-password
   VCD_ORG=your-org-name
   VCD_VDC=your-vdc-name
   VCD_URL=https://your-vcd-instance.example.com/api
   DEFAULT_GATEWAY=default-gateway-ip
   SSH_PUBLIC_KEY="ssh-rsa AAAA... your-key-here"

   # Optional values can be left as defaults
   ```

4. Deploy using the provided script:
   ```bash
   ./scripts/deploy.sh
   ```

   The script will automatically generate terraform.tfvars from your .env file.

## Configuration Variables

### Required Variables (in .env)
- `VCD_PASSWORD`: Your VCD administrator password
- `VCD_ORG`: Your VCD organization name
- `VCD_VDC`: Your VCD virtual datacenter name
- `VCD_URL`: Your VCD API URL
- `DEFAULT_GATEWAY`: Your network gateway IP
- `SSH_PUBLIC_KEY`: Your SSH public key for VM access

### Optional Variables (defaults provided)
- `VCD_USER`: VCD username (default: "administrator")
- `NETWORK_SEGMENT`: Network name (default: "default-network")
- `TEMPLATE_NAME`: VM template name (default: "ubuntu-template")
- `VM_NAME`: Name for the VM (default: "inference-demo-vm")
- `DNS_SERVERS`: Comma-separated DNS servers (default: "8.8.8.8,8.8.4.4")

## API Endpoints

### Generate Text
- **POST** `/generate`
  ```json
  {
    "prompt": "Your text prompt here"
  }
  ```

### Health Check
- **GET** `/health`

## Development

1. Set up local environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r app/requirements.txt
   ```

2. Run locally:
   ```bash
   cd app/src
   python app.py
   ```

## Architecture

- Flask-based API server
- Multiple inference models with CPU optimizations
- Nginx reverse proxy
- Supervisor process management
- Terraform infrastructure as code
- VMware Cloud Director deployment

## License

[MIT License](LICENSE)