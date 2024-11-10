variable "vcd_user" {
  description = "VCD administrator username"
  type        = string
  default     = "administrator"
}

variable "vcd_password" {
  description = "VCD administrator password"
  type        = string
  sensitive   = true
}

variable "vcd_org" {
  description = "VCD organization"
  type        = string
}

variable "vcd_vdc" {
  description = "VCD virtual datacenter"
  type        = string
}

variable "vcd_url" {
  description = "VCD API URL"
  type        = string
}

variable "network_segment" {
  description = "Network segment to connect the VM to"
  type        = string
  default     = "default-network"
}

variable "template_name" {
  description = "Name of the VM template to use"
  type        = string
  default     = "ubuntu-template"
}

variable "vm_name" {
  description = "Name of the VM and vApp to be created"
  type        = string
  default     = "inference-demo"
}

variable "ssh_public_key" {
  description = "SSH public key for VM access"
  type        = string
}

variable "default_gateway" {
  description = "Default gateway for the VM"
  type        = string
}

variable "dns_servers" {
  description = "List of DNS servers"
  type        = list(string)
  default     = ["8.8.8.8", "8.8.4.4"]
}

variable "vm_memory" {
  description = "Memory in MB for the VM"
  type        = number
  default     = 65536  # 64 GB for multiple models
}

variable "vm_cpus" {
  description = "Number of CPUs for the VM"
  type        = number
  default     = 32     # Increased for better inference performance
}

variable "vm_cpu_cores" {
  description = "Number of cores per CPU for the VM"
  type        = number
  default     = 1
}

variable "vm_cpu_hot_add_enabled" {
  description = "Enable CPU hot add"
  type        = bool
  default     = true
}

variable "vm_memory_hot_add_enabled" {
  description = "Enable memory hot add"
  type        = bool
  default     = true
}

variable "vm_catalog_name" {
  description = "Name of the catalog containing the VM template"
  type        = string
  default     = "Local Catalog"
}

variable "vm_ip" {
  description = "Static IP address for the VM"
  type        = string
}

variable "vm_disk_size" {
  description = "Size of the VM's disk in MB"
  type        = number
  default     = 81920  # 80 GB for model storage
}
