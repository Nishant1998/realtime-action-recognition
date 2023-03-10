import time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.sum = None
        self.count = None
        self.avg = None
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class MaxMeter(object):
    """Computes and stores the maximum and current value"""
    def __init__(self):
        self.max = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.max = 0

    def update(self, val):
        self.val = val
        self.max = max(self.max, val)


class Meter(object):
    """Computes and stores the values over the epochs"""
    def __init__(self):
        self.values = None
        self.reset()

    def reset(self):
        self.values = []

    def update(self, val):
        self.values.append(val)


class Counter(object):
    """Keeps track of the number of times an event occurs"""
    def __init__(self):
        self.count = 0

    def reset(self):
        self.count = 0

    def increment(self):
        self.count += 1


class CountDown(object):
    """Keeps track of the number of times an event occurs"""
    def __init__(self, n):
        self.n = n
        self.count = n

    def reset(self):
        self.count = self.n

    def set(self, n):
        self.n = n
        self.count = self.n

    def decrement(self):
        self.count -= 1


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def get_duration(self):
        if self.start_time is None:
            raise ValueError("Timer hasn't been started yet.")
        elif self.end_time is None:
            raise ValueError("Timer hasn't been stopped yet.")
        else:
            return self.end_time - self.start_time
