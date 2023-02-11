docker build -t pipeline .
#docker run --name run_pipeline pipeline
docker run -d --name run_pipeline -it -v "$(pwd)/src:/app/src" pipeline bash #--rm
#docker logs -f run_pipeline
#docker restart run_pipeline
