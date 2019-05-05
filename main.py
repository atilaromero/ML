from arghandler import *

@subcmd
def train(x,y,z):
    print(1,x,y,z)
def predict(self, args):
    print(2)

handler = ArgumentHandler()
handler.run()