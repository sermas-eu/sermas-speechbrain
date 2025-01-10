
FILEPATH?=test.wav
DOCKER_COMPOSE_CMD=docker compose

build:
	$(DOCKER_COMPOSE_CMD) -f docker-compose.local.yaml build

dev:
	$(DOCKER_COMPOSE_CMD) -f docker-compose.local.yaml run --rm speechbrain

serve:
	$(DOCKER_COMPOSE_CMD) -f docker-compose.local.yaml run --rm -p 5011:5011 speechbrain

sh:
	$(DOCKER_COMPOSE_CMD) -f docker-compose.local.yaml run --entrypoint bash --rm speechbrain

req:
	curl --form file='@${FILEPATH}' http://localhost:5011
