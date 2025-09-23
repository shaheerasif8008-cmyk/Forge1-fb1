terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.116.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "location" { type = string, default = "eastus" }
variable "prefix"   { type = string, default = "forge1" }

resource "azurerm_resource_group" "rg" {
  name     = "${var.prefix}-rg"
  location = var.location
}

resource "azurerm_log_analytics_workspace" "law" {
  name                = "${var.prefix}-law"
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = "${var.prefix}-aks"
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "${var.prefix}-dns"

  default_node_pool {
    name       = "system"
    node_count = 3
    vm_size    = "Standard_DS2_v2"
  }

  identity {
    type = "SystemAssigned"
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.law.id
  }
}

resource "azurerm_key_vault" "kv" {
  name                        = "${var.prefix}kv${random_string.suffix.result}"
  location                    = var.location
  resource_group_name         = azurerm_resource_group.rg.name
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  sku_name                    = "standard"
  purge_protection_enabled    = true
  soft_delete_retention_days  = 7
}

resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

data "azurerm_client_config" "current" {}

output "resource_group" { value = azurerm_resource_group.rg.name }
output "aks_name"       { value = azurerm_kubernetes_cluster.aks.name }
output "key_vault"      { value = azurerm_key_vault.kv.name }
output "location"       { value = var.location }

