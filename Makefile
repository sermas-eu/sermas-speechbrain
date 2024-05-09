
FILEPATH?=test.wav

build:
	docker-compose -f docker-compose.local.yaml build

dev:
	docker-compose -f docker-compose.local.yaml run --rm speechbrain

serve:
	docker-compose -f docker-compose.local.yaml run --rm -p 5011:5011 speechbrain

sh:
	docker-compose -f docker-compose.local.yaml run --entrypoint bash --rm speechbrain

req:
	curl --form file='@${FILEPATH}' http://localhost:5011