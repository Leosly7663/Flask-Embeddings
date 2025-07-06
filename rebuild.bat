@echo off
REM Stop and remove the existing container
docker stop flask-docker
docker rm flask-docker

REM Rebuild the Docker image
docker build -t flask-docker .

REM Run the container again
docker run -d --name flask-docker -p 5000:5000 flask-docker

echo Done! The container has been rebuilt and relaunched.

