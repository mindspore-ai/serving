import logging
from typing import Dict, List, Tuple, Set
import asyncio

from .utils import ResponseOutput
from .response_async_queue import AsyncResultsOfOneRequest
from mindspore_serving.serving_utils import PADDING_REQUEST_ID


class RequestEngine:
    def __init__(self):
        self._results_streams: Dict[str, AsyncResultsOfOneRequest] = {}         # "request_id : AsyncResults"
        self._done_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncResultsOfOneRequest, dict]] = asyncio.Queue()

    def process_request_output(self, request_output: ResponseOutput):
        """Add ResponseOutput to Asycnio Queue."""
        
        request_id = request_output.request_id
        if request_id == PADDING_REQUEST_ID:
            logging.debug("process padding request output")
            return
        self._results_streams[request_id].put(request_output)
        if request_output.finished:
            self.abort_request(request_id)
    
    def abort_request(self, request_id: str) -> None:
        self._done_requests.put_nowait(request_id)  
        if request_id not in self._results_streams or \
            self._results_streams[request_id].finished:
            return

        self._results_streams[request_id].finish()
       
    def get_requests_from_register_pool(self) -> Tuple[List[dict], Set[str]]:
        """Get the new requests and finished requests to be sent to the master."""
       
        new_requests: List[dict] = []
        finished_requests: Set[str] = set()
        # finished requests，pop from results_streams
       
        while not self._done_requests.empty():
            request_id = self._done_requests.get_nowait()
            finished_requests.add(request_id)
            self._results_streams[request_id] = AsyncResultsOfOneRequest(request_id)
       
        # get a request from _new_request queue
        while not self._new_requests.empty():
            # request process logic FIFO 
            stream, new_request = self._new_requests.get_nowait()
            # check request status, if finished, put it into finished request set
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._results_streams[stream.request_id] = stream
            new_requests.append(new_request)

        return new_requests, finished_requests

    def register_request(self, request_id: str, **add_request_info) -> AsyncResultsOfOneRequest:
        # Check if the request exists
        if request_id in self._results_streams:
            raise KeyError(f"Request {request_id} already exists.")
        # put a new request into queue(asyncio), and return it's results
        res_stream = AsyncResultsOfOneRequest(request_id)
        self._new_requests.put_nowait((res_stream, {
            "request_id": request_id,
            **add_request_info
        }))
        return res_stream