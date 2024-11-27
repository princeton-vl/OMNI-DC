import torch

class PytocrhCudaTimer:
    def __init__(self, name='', do_timing=True):
        self.name = name
        self.do_timing = do_timing

    def __enter__(self):
        if self.do_timing:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.do_timing:
            self.end.record()
            torch.cuda.synchronize()

            print('Timing: ', self.name, self.start.elapsed_time(self.end))
        else:
            pass
