'''
Buffers for memory-based global temporal shift inference
Global queue: past buffer
Future_buffer: future buffer
Copyright @ Junming Chen
'''
import queue

def _init(future_buffer_len): 
    global _global_queue
    _global_queue = queue.Queue()

    global future_buf_len
    future_buf_len = future_buffer_len

    global batch_index
    batch_index = -1

def _clean(): 
    with _global_queue.mutex:
        _global_queue.queue.clear()

def put(value):
    _global_queue.put(value)


def get():
    return _global_queue.get()

def qsize():
    return _global_queue.qsize()

def get_future_buffer_length():
    return future_buf_len
def set_future_buffer_length(future_buffer_len):
    global future_buf_len
    future_buf_len = future_buffer_len
    return future_buf_len


def get_batch_index():
    return batch_index
def set_batch_index(idx):
    global batch_index
    batch_index = idx

