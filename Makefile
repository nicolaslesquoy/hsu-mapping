PYTHON=python

run:
	$(PYTHON) run.py > run.log 2>&1
	@echo "Run completed. Check run.log for details."

plot:
	$(PYTHON) plot.py

mesh:
	$(PYTHON) mesh.py

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