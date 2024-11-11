#!/bin/bash
set -e

echo "Starting manual setup..."

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install required packages
echo "Installing required packages..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-distutils-extra \
    build-essential \
    nginx \
    supervisor \
    git \
    htop \
    iotop \
    net-tools

# Create app directory structure
echo "Setting up application directory..."
APP_DIR="/home/ubuntu/app"

# Clone the repository
echo "Cloning application repository..."
if [ -d "$APP_DIR/.git" ]; then
    echo "Git repository already exists, pulling latest changes..."
    cd $APP_DIR
    git pull
else
    echo "Cloning fresh repository..."
    git clone https://github.com/arkdna/ark-inference-demo.git $APP_DIR
fi

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
cd $APP_DIR
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "Installing Python dependencies..."
python3 -m pip install -r app/requirements.txt

# Download models
echo "Downloading models..."
python3 scripts/download_models.py

# Configure Nginx
echo "Configuring Nginx..."
sudo tee /etc/nginx/sites-available/inference-demo << EOF
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
    }
}
EOF

# Enable Nginx site
sudo ln -sf /etc/nginx/sites-available/inference-demo /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Configure Supervisor
echo "Configuring Supervisor..."
sudo tee /etc/supervisor/conf.d/inference-demo.conf << EOF
[program:inference-demo]
directory=/home/ubuntu/app
command=/home/ubuntu/app/venv/bin/python3 app/src/app.py
user=ubuntu
autostart=true
autorestart=true
stderr_logfile=/var/log/inference-demo.err.log
stdout_logfile=/var/log/inference-demo.out.log
environment=PATH="/home/ubuntu/app/venv/bin",PYTHONPATH="/home/ubuntu/app"
startsecs=5
stopwaitsecs=5
EOF

# Set permissions
echo "Setting permissions..."
sudo chown -R ubuntu:ubuntu $APP_DIR
sudo chmod 755 $APP_DIR

# Configure system limits
echo "Configuring system limits..."
sudo tee /etc/security/limits.d/inference-limits.conf << EOF
ubuntu soft nofile 65535
ubuntu hard nofile 65535
ubuntu soft nproc 65535
ubuntu hard nproc 65535
EOF

# Restart services
echo "Restarting services..."
sudo systemctl restart nginx
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart inference-demo

# Configure firewall
echo "Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
# Add port 5000 to UFW
sudo ufw allow 5000/tcp
# Verify the rules
sudo ufw status numbered
sudo ufw --force enable

# Final status check
echo "Checking service status..."
sudo systemctl status nginx --no-pager
sudo supervisorctl status

echo "Setup complete! Check the following logs for details:"
echo "- Application logs: tail -f /var/log/inference-demo.{err,out}.log"
echo "- Nginx logs: tail -f /var/log/nginx/error.log"
echo "- Cloud-init log: tail -f /var/log/cloud-init-output.log"

# Print the IP address
echo "Your server IP address is:"
hostname -I | awk '{print $1}'