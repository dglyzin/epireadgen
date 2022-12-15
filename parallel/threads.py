import queue
import threading
import multiprocessing as mp
import time
# import sympy


def par_wrap(items, f, *args, **kwargs):
    '''
    solve items using all available system threads
    return merged result
    '''
    num_worker_threads = mp.cpu_count()
    print("count of physical cpu:")
    print(num_worker_threads)

    q = queue.Queue()

    threads = []

    for i in range(num_worker_threads):
        # t = threading.Thread(target=worker)
        t = Kernel(q, i)
        t.set_work(f, *args, **kwargs)
        t.start()
        threads.append(t)

    print("beginning to adding a work")
    time_start = time.time()
    # for item in [[1, 2, 3], [4, 5, 6]]:
    for item in items:
        q.put(item)
    print("end of adding a work")
    # block until all tasks are done
    q.join()

    # stop workers
    for i in range(num_worker_threads):
        q.put(None)
    for t in threads:
        t.join()

    results = []
    for thread in threads:
        results.extend(thread.results)

    print("unblock")
    print("running time:")
    print(time.time()-time_start)
    return results

    
class Kernel(threading.Thread):
    def __init__(self, work_queue, number):
        threading.Thread.__init__(self)
        self.work_queue = work_queue
        
        self.number = number
        self.work = None
        
    def set_work(self, work, *args, **kwargs):
        self.work = work
        self.work_common_args = args
        self.work_common_kwargs = kwargs

    def run(self):
        
        self.results = []
        while True:
            try:
                # entry here means one seq from the queue: 
                entry = self.work_queue.get()
                if entry is None:
                    break
                if self.work is not None:
                    # if use this result as self.result
                    # then the error "Kernel obj has no result"
                    # will be generate since the break statement above:
                    result = self.work(
                        entry, *self.work_common_args,
                        **self.work_common_kwargs)
                    self.results.append(result)
                else:
                    self.do_work_default(entry)
                    
            finally:
                self.work_queue.task_done()
                
    def do_work_default(self, entry):
        '''What will be done for each thread'''
        print("thread %s " % (str(self.number)), "entry:")
        print(entry)
        for e in entry:
            print((lambda x: eval("pow(x, 3)"))(e))
        # print((lambda x: eval("sympy.sin(x)"))(1))


def test_par_wrap():
    items = [["red", "green", "blue"],
             ["second red", "sgreen", "sblue"]]
    common_args = ("colors support is ",)

    def f(vals, var):
        print(var, vals)
        return (var, vals)
    res = par_wrap(items, f, *common_args)

    print("par_wrap result:")
    print(res)


def test_basic():

    print("count of physical cpu:")
    print(mp.cpu_count())

    num_worker_threads = 2

    q = queue.Queue()

    threads = []

    for i in range(num_worker_threads):
        # t = threading.Thread(target=worker)
        t = Kernel(q, i)
        t.start()
        threads.append(t)

    # part 1:
    print("beginning to adding a work")
    time_start = time.time()
    for item in [[1, 2, 3], [4, 5, 6]]:
        q.put(item)
    print("end of adding a work")
    # block until all tasks are done
    q.join()
    print("unblock")
    print("running time:")
    print(time.time()-time_start)

    # part 2:
    print("beginning to addding a work")
    time_start = time.time()
    for item in [[1, 2, 3], [4, 5, 6]]:
        q.put(item)
    print("end of add work")
    # block until all tasks are done
    q.join()
    print("unblock")
    print("running time:")
    print(time.time()-time_start)
    
    # stop workers
    for i in range(num_worker_threads):
        q.put(None)
    for t in threads:
        t.join()


if __name__ == "__main__":
    test_par_wrap()
    # test_basic()
