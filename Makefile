.PHONY: install test run cli-predict docker-build docker-run clean

install:
	pip install -r requirements.txt
	pnpm install

test:
	pytest tests/ -v
	pnpm test

run:
	python app/server.py

cli-predict:
	python cli.py predict -i data/sample_student.json

cli-batch:
	python cli.py predict-batch -i data/sample_new_students.csv -o evaluated_students.csv

docker-build:
	docker build -t ishraf-student-risk-service:latest .

docker-run:
	docker run -p 8000:8000 ishraf-student-risk-service:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
