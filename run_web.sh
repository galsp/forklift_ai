#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "Virtual environment .venv tidak ditemukan. Menjalankan setup..."
    python3 -m venv --system-site-packages .venv
    source .venv/bin/activate
    pip install -r requirements.jetson.txt
else
    source .venv/bin/activate
fi

echo "=========================================================="
echo " Starting Forklift Vision AI Web Server"
echo " Akses Web UI di: http://172.22.20.25:8080"
echo "=========================================================="
exec python web_forklift.py --host 0.0.0.0 --port 8080 "$@"
