import os
from os.path import expanduser
from typing import NamedTuple

class Paths(NamedTuple):
    server_data_path = os.path.join(expanduser('~'), 'project', 'Datasets', 'iframe_720x1280')
    root = os.getcwd()
    data:str = os.path.join(root, 'data')
    model:str = os.path.join(data, 'model')
    result:str = os.path.join(data, 'result')
    dataset:str = os.path.join(data, 'dataset')

    @staticmethod
    def create_dir(path:str):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def init_path():
        for path in Paths():
            Paths.create_dir(path)



if __name__ == '__main__':
    Paths.init_path()