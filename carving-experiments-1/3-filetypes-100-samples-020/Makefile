.PHONY: all clean
SHELL:=/bin/bash

all:
	(time python3 main.py) 2>&1 | tee >(stdbuf -o 0 sed -e 's/.*\r//g' > run.log)

tboard:
	mkdir model.tboard/run0
	sh -c 'mv model.tboard/events* model.tboard/run0/'
	(time python3 main.py) 2>&1 | tee >(stdbuf -o 0 sed -e 's/.*\r//g' > run1.log)
	mkdir model.tboard/run1
	sh -c 'mv model.tboard/events* model.tboard/run1/'
	(time python3 main.py) 2>&1 | tee >(stdbuf -o 0 sed -e 's/.*\r//g' > run2.log)
	mkdir model.tboard/run2
	sh -c 'mv model.tboard/events* model.tboard/run2/'
	(time python3 main.py) 2>&1 | tee >(stdbuf -o 0 sed -e 's/.*\r//g' > run3.log)
	mkdir model.tboard/run3
	sh -c 'mv model.tboard/events* model.tboard/run3/'
	(time python3 main.py) 2>&1 | tee >(stdbuf -o 0 sed -e 's/.*\r//g' > run4.log)
	mkdir model.tboard/run4
	sh -c 'mv model.tboard/events* model.tboard/run4/'

clean:
	sh -c 'rm -r *.h5 *.tboard history.png run.log' || echo
