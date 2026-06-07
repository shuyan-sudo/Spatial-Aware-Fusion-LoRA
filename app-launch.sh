#!/usr/bin/env bash

source /etc/network_turbo
cd "`dirname "$0"`" || exit 1
. env/bin/activate
export MPLBACKEND=agg  # 设置 MPLBACKEND 环境变量为合适的后端
python app.py
