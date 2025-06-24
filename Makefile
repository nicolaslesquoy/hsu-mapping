PYTHON=$(HOME)/.venv/bin/python

run:
	$(PYTHON) run.py > run.log 2>&1
	@echo "Run completed. Check run.log for details."

mesh:
	$(PYTHON) mesh.py

batch:
	$(PYTHON) batch.py

optim:
	$(PYTHON) optim.py

optim_gpu: obj/optim.o
    nvcc obj/optim.o -o bin/optim_gpu

obj/optim.o: optim.cu
    mkdir -p obj
    nvcc -c optim.cu -o obj/optim.o

test:
	$(PYTHON) test.py

lint:
	ruff check *.py
	@echo "Linting completed."

format:
	ruff format *.py
	@echo "Formatting completed."

clean:
	rm -rf a_mesh b_mesh mapped
	rm -rf precice-profiling precice-run
	rm precice-config.xml input_mesh.vtu
	@echo "Cleaned up temporary files."

clean-fig:
	rm ./*.pdf