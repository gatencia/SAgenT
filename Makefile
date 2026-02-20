.PHONY: build up down shell bench setup logs

# Default values for environment variables if not set
SERVICE_NAME=app

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose run --rm $(SERVICE_NAME) bash

setup:
	bash set_up.sh

bench:
	docker compose run --rm $(SERVICE_NAME) python bench/run_bench.py $(ARGS)

logs:
	docker compose logs -f $(SERVICE_NAME)

# Example: make bench ARGS="--id mrpp_8x8_4r_T20_sat_central_block --provider google"
