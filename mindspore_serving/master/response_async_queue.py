from queue import Queue
import asyncio
from .utils import Counter, ResponseOutput


class AsyncResultsOfOneRequest:
    def __init__(self, request_id: str):

        self.request_id = request_id
        self._queue:asyncio.Queue[ResponseOutput] = asyncio.Queue()    # using to save inference result(a token and it's decode text) 
        self._finished = False          

    def put(self, item: ResponseOutput) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopIteration)
        self._finished = True
    
    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> ResponseOutput:
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        elif isinstance(result, Exception):
            raise result
        return result