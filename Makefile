PYTHON=/bin/python3

run:
	$(PYTHON) run.py

plot:
	$(PYTHON) plot.py

mesh:
	$(PYTHON) mesh.py

test:
	$(PYTHON) test.py

format:
	black run.py plot.py

clean:
	rm -rf a_mesh b_mesh mapped
	rm -rf precice-profiling precice-run
	rm input_mesh.vtu
	rm -rf result*