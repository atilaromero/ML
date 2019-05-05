from arghandler import *
import os

@subcmd
def train(x,y,z):
    for dirpath, dirnames, filenames in os.walk('carving/dataset'):
        for f in filenames:
            yield dirpath, f
def predict(self, args):
    print(2)

handler = ArgumentHandler()
handler.run()