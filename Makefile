.PHONY: dev dev-cuda server pipeline pipeline-cuda frontend

# Run all services (CPU)
dev:
	@echo "Starting server, pipeline, and frontend..."
	$(MAKE) -j3 server pipeline frontend

# Run all services (CUDA)
dev-cuda:
	@echo "Starting server, pipeline (CUDA), and frontend..."
	$(MAKE) -j3 server pipeline-cuda frontend

server:
	cd server && pixi run serve

pipeline:
	cd pipeline && pixi run worker

pipeline-cuda:
	cd pipeline && pixi run -e cuda worker

frontend:
	npm run dev
