from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor


def initRKNN(rknnModel=r"best_nano_111_rv1126b_fp.rknn", id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("LoadRKNNrknnModel failed")
        exit(ret)
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtimeenvironment failed")
        exit(ret)
    print(rknnModel, "\t\tdone")
    return rknn_lite


def initRKNNs(rknnModel="best_nano_111_rv1126b_fp.rknn", TPEs=1):
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 2))
    return rknn_list


from collections import deque

class rknnPoolExecutor():
    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = TPEs
        self.rknnPool = initRKNNs(rknnModel, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0
        self.futures = deque()  # 多任务队列

    def put(self, frame):
        if len(self.futures) >= self.TPEs:
            # 队列满，丢帧或者等待，可根据策略改
            return False
        rknn = self.rknnPool[self.num % self.TPEs]
        future = self.pool.submit(self.func, rknn, frame)
        self.futures.append(future)
        self.num += 1
        return True

    def get(self):
        if len(self.futures) == 0:
            return None, False
        future = self.futures[0]
        if not future.done():
            return None, False
        self.futures.popleft()
        result = future.result()
        return result, True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()