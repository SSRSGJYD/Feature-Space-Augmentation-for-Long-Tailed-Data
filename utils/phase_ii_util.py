import multiprocessing


def multiprocess_in_sequence(func, param_list, workers=10):
    param_data = [[] for _ in range(workers)]
    c_worker = 0
    for i, param in enumerate(param_list):
        param_data[c_worker].append((i, param))
        c_worker = (c_worker + 1) % workers

    q = multiprocessing.Queue()
    q.cancel_join_thread()
    count = 0

    def worker(func, param_part_list):
        for i, param in param_part_list:
            try:
                if isinstance(param, list):
                    data = func(*param)
                else:
                    data = func(param)
            except:
                continue
            q.put((i, data))

    for i in range(workers):
        w = multiprocessing.Process(
            target=worker,
            args=(func, param_data[i]))
        w.daemon = False
        w.start()

    data_list = [None for _ in range(len(param_list))]

    while count < len(param_list):
        i, data = q.get()
        data_list[i] = data
        count += 1

    return data_list
