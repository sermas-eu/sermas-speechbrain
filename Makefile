
FILEPATH?=test.wav
DOCKER_COMPOSE_CMD=docker compose -f docker-compose.yaml

build:
	$(DOCKER_COMPOSE_CMD) build

stop:
	$(DOCKER_COMPOSE_CMD) kill speechbrain || true
	$(DOCKER_COMPOSE_CMD) rm -f speechbrain || true

start: stop
	$(DOCKER_COMPOSE_CMD) up speechbrain

sh:
	$(DOCKER_COMPOSE_CMD) run --entrypoint bash --rm speechbrain

req:
	curl --form file='@${FILEPATH}' http://localhost:5011

setup:
	python3 -m venv .venv
	./.venv/bin/pip3 install -r requirements.txt

push:
	docker tag sermas/speechbrain ghcr.io/sermas-eu/speechbrain:test
	docker push ghcr.io/sermas-eu/speechbrain:test
