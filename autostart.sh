#!/bin/bash

# Optional: wait a bit to ensure networking is up
sleep 10

# Navigate to your app directoryC:\Users\Leonardo Nigro\Desktop\Docker Projects\Flask Docker\app.py
cd "C:/Users/Leonardo Nigro/Desktop/Docker Projects/Flask Docker"

# Start the docker container
docker start flask-docker || docker run -d --name flask-docker -p 5000:5000 flask-docker

# Start cloudflared tunnel
cloudflared tunnel run flask-docker
