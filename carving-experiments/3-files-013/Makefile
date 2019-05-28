.PHONY: all clean
SHELL := /bin/bash

all:
	(time python3 main.py) |& tee run.log

clean:
	sh -c 'rm -r *.h5 *.tboard history.png run.log'