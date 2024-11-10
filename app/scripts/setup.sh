#!/bin/bash

# Update and install system packages
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    nginx \
    supervisor

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure nginx
sudo cp config/nginx/inference-demo /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/inference-demo /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Configure supervisor
sudo cp config/supervisor/inference-demo.conf /etc/supervisor/conf.d/

# Restart services
sudo systemctl restart nginx
sudo systemctl enable supervisor
sudo systemctl restart supervisor

# Configure firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw --force enable