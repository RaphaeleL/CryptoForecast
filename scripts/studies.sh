# !/bin/bash

set -xe

./scripts/prepare.sh

python3 main.py -c LTC-EUR -s -d 4
python3 main.py -c BTC-EUR -s -d 4
python3 main.py -c ETH-EUR -s -d 2