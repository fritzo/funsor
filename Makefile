all: test

install:
	pip install -e .[dev]

lint: FORCE
	flake8

format: FORCE
	isort -y

test: lint FORCE
	pytest -v test
	FUNSOR_DEBUG=1 pytest -v test/test_gaussian.py
	python examples/discrete_hmm.py -n 2
	python examples/discrete_hmm.py -n 2 -t 50 --lazy
	python examples/kalman_filter.py --xfail-if-not-implemented
	python examples/kalman_filter.py -n 2 -t 50 --lazy
	python examples/pcfg.py --size 3
	python examples/vae.py --smoke-test
	@#python examples/ss_vae_delayed.py --xfail-if-not-implemented
	@#python examples/minipyro.py --xfail-if-not-implemented
	@echo PASS

clean: FORCE
	git clean -dfx -e pyro-egg.info

FORCE:
