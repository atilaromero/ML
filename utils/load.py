import os

def category_from_extension(path):
    return path.rsplit('.',1)[1]

def category_from_folder(path):
    return path.rsplit('/',2)[-2]

def examples_from(folder):
    def inner():
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                yield os.path.join(dirpath, f)
    return list(inner())
