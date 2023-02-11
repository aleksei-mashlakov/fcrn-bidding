#!/bin/sh
git pull origin master
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d
#docker-compose run -d --name run_pipeline pipeline
#docker logs -f run_pipeline
# docker-compose down
