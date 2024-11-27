from importlib import import_module

def get(args):
    metric_name = args.model_name + 'Metric'
    module_name = 'metric.' + metric_name.lower()
    module = import_module(module_name)

    return getattr(module, metric_name)

class BaseMetric:
    def __init__(self, args):
        self.args = args

    def evaluate(self, sample, output, mode):
        pass
