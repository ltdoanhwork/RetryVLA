#!/usr/bin/env bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; uptime; top -bn1 | head -20 | tail -12
