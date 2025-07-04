#!/bin/bash
apt-get update
apt-get install -y default-jre
streamlit run app.py --server.port=10000 --server.address=0.0.0.0
