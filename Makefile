PYTHON=python

run:
	$(PYTHON) run.py > run.log 2>&1
	@echo "Run completed. Check run.log for details."

mesh:
	$(PYTHON) mesh.py

batch:
	$(PYTHON) batch.py

lint:
	ruff check *.py
	@echo "Linting completed."

format:
	ruff format *.py
	@echo "Formatting completed."

clean:
	rm -rf a_mesh b_mesh mapped
	rm -rf precice-profiling precice-run
	@echo "Cleaned up temporary files."

clean-fig:
	rm ./*.pdf