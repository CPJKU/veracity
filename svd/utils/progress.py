import sys
import time


def progress(items, desc='', total=None, min_delay=0.1):
    """ Progress indicator for iterator;
        taken from https://github.com/f0k/ismir2015/blob/master/experiments/progress.py. """
    total = total or len(items)
    t_start = time.time()
    t_last = 0
    for n, item in enumerate(items):
        t_now = time.time()
        if t_now - t_last > min_delay:
            print('\r%s%d/%d (%6.2f%%)' % (desc, n+1, total, n / float(total) * 100), end=' ')
            if n > 0:
                t_done = t_now - t_start
                t_total = t_done / n * total
                print('(ETA: %d:%02d)' % divmod(t_total - t_done, 60), end=' ')
            sys.stdout.flush()
            t_last = t_now
        yield item
    t_total = time.time() - t_start
    print('\r%s%d/%d (100.00%%) (took %d:%02d)' % ((desc, total, total) + divmod(t_total, 60)))


def progress_data(data, desc='', total=None, min_delay=0.1):
    """ Progress indicator for iterative data set. """
    t_start = time.time()
    t_last = 0
    n = 0
    for d in data:
        if n == total:
            break
        t_now = time.time()
        if t_now - t_last > min_delay:
            print('\r%s%d/%d (%6.2f%%)' % (desc, n+1, total, n / float(total) * 100), end=' ')
            if n > 0:
                t_done = t_now - t_start
                t_total = t_done / n * total
                print('(ETA: %d:%02d)' % divmod(t_total - t_done, 60), end=' ')
            sys.stdout.flush()
            t_last = t_now
        n += 1
        yield d
    t_total = time.time() - t_start
    print('\r%s%d/%d (100.00%%) (took %d:%02d)' % ((desc, total, total) + divmod(t_total, 60)))
