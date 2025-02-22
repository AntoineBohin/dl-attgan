import os
from glob import glob

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def name_experiment(prefix='', suffix=''):
    import datetime
    import platform
    
    experiment_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + platform.node()
    if prefix is not None and prefix != '':
        experiment_name = prefix + '_' + experiment_name
    if suffix is not None and suffix != '':
        experiment_name = experiment_name + '_' + suffix
    return experiment_name


def find_model(path, epoch='latest'):
    if epoch == 'latest':
        files = glob(os.path.join(path, '*.pth'))
        file = sorted(files, key=lambda x: int(x.rsplit('.', 2)[1]))[-1]
    else:
        file = os.path.join(path, 'weights.{:d}.pth'.format(int(epoch)))
    assert os.path.exists(file), 'File not found: ' + file
    print('Find model of {} epoch: {}'.format(epoch, file))
    return file


class Progressbar():
    def __init__(self):
        self.p = None
    def __call__(self, iterable):
        from tqdm import tqdm
        self.p = tqdm(iterable)
        return self.p
    def say(self, **kwargs):
        if self.p is not None:
            self.p.set_postfix(**kwargs)

def add_scalar_dict(writer, scalar_dict, iteration, directory=None):
    for key in scalar_dict:
        key_ = directory + '/' + key if directory is not None else key
        writer.add_scalar(key_, scalar_dict[key], iteration)