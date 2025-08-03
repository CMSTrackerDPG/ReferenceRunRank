#!/bin/bash

echo "Creating a virtual environment..."
python3.11 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Please check your Python installation."
    exit 1
fi

echo "Activating the virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment. Please check your shell configuration."
    exit 1
fi

echo "Installing required packages..."
pip3 install .