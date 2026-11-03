"""Overlap one capture producer with one consumer, with explicit ownership.

At depth two the next request can run while the current record is decoded and
persisted. No third request is submitted until the current consumer completes.
The producer owns cleanup until it returns; the consumer owns cleanup from its
call onward. Unconsumed producer results are discarded on failure or close.
"""

from concurrent.futures import ThreadPoolExecutor


def capture_pipeline(items, *, produce, consume, discard, depth=1):
    if depth not in (1, 2):
        raise ValueError("capture pipeline depth must be 1 or 2")
    if depth == 1:
        for item in items:
            yield consume(item, produce(item))
        return
    source = iter(items)
    sentinel = object()
    item = next(source, sentinel)
    if item is sentinel:
        return
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="capture-producer"
    ) as pool:
        future = pool.submit(produce, item)
        try:
            while future is not None:
                current = item
                result = future.result()
                future = None
                try:
                    item = next(source, sentinel)
                    if item is not sentinel:
                        future = pool.submit(produce, item)
                except BaseException:
                    discard(current, result)
                    raise
                yield consume(current, result)
        finally:
            if future is not None and not future.cancel():
                # Wait for the sole in-flight request before closing the store.
                # Its producer cleans up partial failures itself.
                try:
                    result = future.result()
                except Exception:
                    pass
                else:
                    discard(item, result)
