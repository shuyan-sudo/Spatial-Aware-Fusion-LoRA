#!/usr/bin/env bash

source /etc/network_turbo
cd "`dirname "$0"`" || exit 1
. env/bin/activate
export MPLBACKEND=agg  # Set the MPLBACKEND environment variable to a suitable backend
python app.py
