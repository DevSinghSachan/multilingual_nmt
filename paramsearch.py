import os
import sys
from multiprocessing import Process, Queue
import subprocess
import itertools

num_gpus = int(sys.argv[1])


def work(id, queue):
    gpu_id = str(id)

    while True:
        arg_val = queue.get()

        if arg_val is None:
            break

        arg_string = \
        'env CUDA_VISIBLE_DEVICES={} bash tools/bpe_pipeline_en_vi.sh {} {} {} {} {}'

        cmd_string = arg_string.format(gpu_id,
                                       'Yogi',
                                       arg_val[0],
                                       arg_val[1],
                                       arg_val[2],
                                       arg_val[3])

        print(cmd_string)
        my_env = os.environ.copy()
        my_env['CUDA_VISIBLE_DEVICES'] = gpu_id
        subprocess.call(cmd_string.split(), shell=False)

    queue.put(None)


lr = [1e-4, 5e-4]
beta1 = [0.9]
beta2 = [0.9, 0.99, 0.999]
eps = [1e-3, 1e-8]

def serve(queue):
    for tuple_ in itertools.product(lr, beta1, beta2, eps):
        queue.put(tuple_)


# https://stackoverflow.com/questions/914821/producer-consumer-problem-with-python-multiprocessing
class Manager:
    def __init__(self):
        self.queue = Queue()
        self.num_process = num_gpus

    def start(self):
        print("starting %d workers" % self.num_process)
        self.workers = [Process(target=work, args=(i, self.queue,))
                        for i in range(self.num_process)]
        for w in self.workers:
            w.start()
        serve(self.queue)

    def stop(self):
        self.queue.put(None)
        for i in range(self.num_process):
            self.workers[i].join()
        self.queue.close()


if __name__ == '__main__':
    m = Manager()
    m.start()
    m.stop()