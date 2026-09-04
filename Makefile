.PHONY: service pipeline probe profile compile eval optimize migrate

service-run:
	cd apc-service && mvn spring-boot:run

service-build:
	cd apc-service && mvn clean package -DskipTests

pipeline-install:
	cd apc-pipeline && pip install -e ".[dev]"

probe:
	cd apc-pipeline && apc probe run --model glm --suite v1

profile:
	cd apc-pipeline && apc profile build --model glm

compile:
	cd apc-pipeline && apc prompt compile --task ../configs/tasks/financial_analysis.yaml --model glm --genome ../configs/genomes/base.json

optimize:
	cd apc-pipeline && apc optimize run --task financial_analysis_v1 --model glm --budget 100
