#cloud-config
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - ${ssh_public_key}

package_update: true
package_upgrade: true

packages:
  - python3
  - python3-pip
  - python3-venv
  - python3-setuptools
  - python3-distutils-extra
  - build-essential
  - nginx
  - supervisor
  - git
  - htop
  - iotop
  - net-tools

write_files:
  - path: /etc/sysctl.d/99-inference-tuning.conf
    content: |
      # VM memory management tuning
      vm.swappiness=10
      vm.dirty_ratio=40
      vm.dirty_background_ratio=10
      
      # File system tuning
      fs.file-max=2097152
      
      # Network tuning
      net.core.somaxconn=65535
      net.ipv4.tcp_max_syn_backlog=65536

  - path: /etc/security/limits.d/inference-limits.conf
    content: |
      ubuntu soft nofile 65535
      ubuntu hard nofile 65535
      ubuntu soft nproc 65535
      ubuntu hard nproc 65535

  - path: /etc/nginx/sites-available/inference-demo
    content: |
      server {
          listen 80 default_server;
          listen [::]:80 default_server;
          server_name _;

          location / {
              proxy_pass http://127.0.0.1:5000;
              proxy_set_header Host $host;
              proxy_set_header X-Real-IP $remote_addr;
              proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
              proxy_set_header X-Forwarded-Proto $scheme;
              proxy_read_timeout 300;
              proxy_connect_timeout 300;
          }
      }

  - path: /etc/supervisor/conf.d/inference-demo.conf
    content: |
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

runcmd:
  # System optimizations
  - sysctl -p /etc/sysctl.d/99-inference-tuning.conf
  - mkdir -p /home/ubuntu/.cache/huggingface
  - chown -R ubuntu:ubuntu /home/ubuntu/.cache

  # Clone and setup application
  - su - ubuntu -c "git clone https://github.com/arkdna/ark-inference-demo.git /home/ubuntu/app"
  - su - ubuntu -c "mkdir -p /home/ubuntu/app/app/{src,static,templates}"
  
  # Python environment setup
  - su - ubuntu -c "cd /home/ubuntu/app && python3 -m venv venv"
  - su - ubuntu -c "cd /home/ubuntu/app && . venv/bin/activate && pip install --upgrade pip setuptools wheel"
  - su - ubuntu -c "cd /home/ubuntu/app && . venv/bin/activate && pip install -r app/requirements.txt"
  
  # Configure services
  - ln -sf /etc/nginx/sites-available/inference-demo /etc/nginx/sites-enabled/
  - rm -f /etc/nginx/sites-enabled/default
  
  # Set permissions
  - chown -R ubuntu:ubuntu /home/ubuntu/app
  - chmod 755 /home/ubuntu/app
  
  # Start services
  - systemctl restart nginx
  - systemctl enable supervisor
  - systemctl restart supervisor
  
  # Configure firewall
  - ufw allow 22/tcp
  - ufw allow 80/tcp
  - ufw --force enable