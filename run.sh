#!/bin/bash

VENV_DIR=.venv

sudo apt-get install python3.10 python3.10-venv 

# Use Python 3.10 temporarily
if [ ! -f /usr/bin/python3 ]; then
    sudo ln -s /usr/bin/python3.10 /usr/bin/python3
fi

# Use pip3.10 temporarily
if [ ! -f /usr/bin/pip3 ]; then
    sudo ln -s /usr/bin/pip3.10 /usr/bin/pip3
fi

# Use python3.10 in the virtual environment
if [ ! -f $VENV_DIR/bin/python ]; then
    ln -s /usr/bin/python3.10 $VENV_DIR/bin/python
fi

# Use pip3.10 in the virtual environment
if [ ! -f $VENV_DIR/bin/pip ]; then
    ln -s /usr/bin/pip3.10 $VENV_DIR/bin/pip
fi

# Check if .venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install specific packages
echo "Installing required packages..."
pip install xformers==0.0.26post1 torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install tkinter
echo "Installing tkinter..."
sudo apt-get install python3.10-tk

# Install additional requirements
if [ -f requirements.txt ]; then
    echo "Installing additional requirements..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found, skipping..."
fi

# Launch the script
echo "Launching LightDiffusion-Flux..."
python3.10 flux.py

# Deactivate the virtual environment
deactivate