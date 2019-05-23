.PHONY: all clean

all:
	time python3 main.py 2>&1 | tee run.log

clean:
	sh -c 'rm -r *.h5 *.tboard history.png run.log'