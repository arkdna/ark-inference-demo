#!/bin/bash

if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Required variables
REQUIRED_VARS=(
    "VCD_PASSWORD"
    "VCD_ORG"
    "VCD_VDC"
    "VCD_URL"
    "DEFAULT_GATEWAY"
    "SSH_PUBLIC_KEY"
)

# Check for missing required variables
MISSING_VARS=0
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        echo "Error: Required variable $VAR is not set in .env"
        MISSING_VARS=1
    fi
done

if [ $MISSING_VARS -eq 1 ]; then
    exit 1
fi 