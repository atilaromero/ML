import sys

class ML:
    def __init__(self, get_model, data_generator, save_file='', **compile_kwargs):
        self.model = get_model()
        self.compile_kwargs = compile_kwargs
        self.data_generator = data_generator
        self.save_file = save_file
        self.commands = ['train']

        if save_file:
            try:
                self.model.load_weights(save_file)
            except OSError:
                pass

    def set_compile_kwargs(self, **compile_kwargs)
        self.compile_kwargs = compile_kwargs

    def set_fit_kwargs(self, **fit_kwargs)
        self.fit_kwargs = fit_kwargs

    def train(self):
        self.model.compile(**self.compile_kwargs)
        self.model.summary()

        for xs, ys in self.data_generator():
            self.model.fit(xs,ys,**self.fit_kwargs)
            self.model.save(self.save_file)

    def main(self, *args):
        if len(args)<2 or not args[1] in self.commands:
            print(f"""Use {sys.argv[1]} COMMAND
            COMMANDS: {self.commands}""")
            print(args)
            exit(1)
        f = getattr(self, args[1])
        f(*args[2:])
