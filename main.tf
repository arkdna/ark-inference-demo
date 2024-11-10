terraform {
  required_version = ">= 0.13"
  required_providers {
    vcd = {
      source  = "vmware/vcd"
      version = "~> 3.8.2"
    }
  }
}

provider "vcd" {
  user                 = var.vcd_user
  password             = var.vcd_password
  auth_type           = "integrated"
  org                 = var.vcd_org
  vdc                 = var.vcd_vdc
  url                 = var.vcd_url
  allow_unverified_ssl = true
}

# Create a vApp
resource "vcd_vapp" "vapp" {
  name        = var.vm_name
  description = "Private Cloud Inference Demo"
}

# Connect vApp to org
resource "vcd_vapp_org_network" "vappnet" {
  vapp_name        = vcd_vapp.vapp.name
  org_network_name = var.network_segment
}

# Create the VM within the vApp
resource "vcd_vapp_vm" "vm" {
  vapp_name     = vcd_vapp.vapp.name
  name          = var.vm_name
  computer_name = var.vm_name
  
  # Hardware configuration
  memory                 = var.vm_memory
  cpus                   = var.vm_cpus
  cpu_cores              = var.vm_cpu_cores
  cpu_hot_add_enabled    = var.vm_cpu_hot_add_enabled
  memory_hot_add_enabled = var.vm_memory_hot_add_enabled

  override_template_disk {
    bus_type        = "paravirtual"
    size_in_mb      = var.vm_disk_size
    bus_number      = 0
    unit_number     = 0
  }

  # OS Template
  catalog_name  = var.vm_catalog_name
  template_name = var.template_name
  
  # Network configuration
  network {
    type               = "org"
    name               = var.network_segment
    ip_allocation_mode = "MANUAL"
    ip                = var.vm_ip
    adapter_type      = "VMXNET3"
    connected         = true
  }

  # Cloud-init configuration
  guest_properties = {
    "guestinfo.userdata" = base64encode(templatefile("cloud-init.tpl", {
      ssh_public_key = var.ssh_public_key
    }))
    "guestinfo.userdata.encoding" = "base64"
  }
}

# Outputs for easy reference
output "vm_internal_ip" {
  description = "Internal IP address of the VM"
  value       = vcd_vapp_vm.vm.network[0].ip
}

output "vm_name" {
  description = "Name of the deployed VM"
  value       = vcd_vapp_vm.vm.name
}

output "vapp_name" {
  description = "Name of the vApp"
  value       = vcd_vapp.vapp.name
}

output "vm_ip" {
  description = "The IP address assigned to the VM"
  value       = vcd_vapp_vm.vm.network[0].ip
}
