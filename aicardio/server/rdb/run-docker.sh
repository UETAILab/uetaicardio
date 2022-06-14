mkdir -p database
docker run \
    -d --name echo-redis-db \
    -p 6333:6379 \
    -v database:/data \
    redis redis-server --appendonly yes
