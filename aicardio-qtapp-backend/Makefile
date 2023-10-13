all: build run-docker

build-plain:
	echo "Build docker image"
	DOCKER_BUILDKIT=1 docker build --progress=plain --build-arg CACHEBUST=$(date +%s) -t echodeploy .

build:
	echo "Build docker image"
	DOCKER_BUILDKIT=1 docker build --build-arg CACHEBUST=$(date +%s) -t echodeploy .

run-docker:
	echo "Run docker image"
	docker run -it --gpus 0 echodeploy:latest

compose-up:
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose up

compose-build:
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose up --build
