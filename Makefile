DOCKER_COMPOSE_CMD=docker compose -f docker-compose.yaml
TORCH_INDEX?=https://download.pytorch.org/whl/cpu

build:
	$(DOCKER_COMPOSE_CMD) build

stop:
	$(DOCKER_COMPOSE_CMD) kill speechbrain || true
	$(DOCKER_COMPOSE_CMD) rm -f speechbrain || true

start: stop
	$(DOCKER_COMPOSE_CMD) up speechbrain

sh:
	$(DOCKER_COMPOSE_CMD) run --entrypoint bash --rm speechbrain

# the test suite inside the image (models are downloaded to ./data/cache on first run)
test:
	$(DOCKER_COMPOSE_CMD) run --rm --no-deps speechbrain python3 -m unittest discover -s tests

setup:
	python3 -m venv .venv
	./.venv/bin/pip3 install --index-url $(TORCH_INDEX) --extra-index-url https://pypi.org/simple -c constraints.txt torch==2.11.0 torchaudio==2.11.0
	./.venv/bin/pip3 install -r requirements.txt -c constraints.txt
