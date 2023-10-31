
import asyncio
from typing import Optional
import logging

from master.master import AsyncMaster
from master.response_async_queue import AsyncResultsOfOneRequest
from master.utils import ResponseOutput
from master.request_resister_engine import RequestEngine



class LLMServer:
     """
        request_queue(FIFO): add request into a async queue, and monitor request status(is_finished), 
                             mapping inference result of each iteration to corresponding request
                             result queue(used in stream return).
        master: Continuously getting unfinished request from request_queue, conducting batch strategy,
                and doing one step inference using ms-lite, after get result of one iteration, client
                get stream inference result from request_queue, and update to request_queue.
     """
     def __init__(self):
        self.request_engine = RequestEngine()   
        self.background_loop = None
        self.master = AsyncMaster()   
        self.background_loop = None
    
     @property
     def is_running(self) -> bool:
        return self.background_loop is not None
        
     async def run_loop(self):
        while True:
            await self.step()
            await asyncio.sleep(0)
         
     def start_background_loop(self) -> None:
        # todo
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self.background_loop = asyncio.get_event_loop().create_task(self.run_loop())

     async def register_request(
        self,
        request_id: str,
        **add_request_info) -> AsyncResultsOfOneRequest:

        if self.background_loop is None:
            self.start_background_loop()
       
        res_stream = self.request_engine.register_request(
            request_id,
            **add_request_info)

        return res_stream

     def _abort(self, request_id: str) -> None:
        """Abort a request.
        Args:
            request_id: The unique id of the request.
        """
        self.request_engine.abort_request(request_id)
    
     async def step(self):
        # loop consuming from request_engine
        new_requests, finished_requests = self.request_engine.get_requests_from_register_pool()
       
        for new_request in new_requests:
            logging.info("new request is {}".format(new_request))
            self.master.add_requests_to_schedule_pool(**new_request)
        
        if finished_requests:
            await self._master_abort(finished_requests)
        request_outputs = await self.master.step_async()
       
        # Put the outputs into the corresponding streams.
        if request_outputs is not None: 
            for request_output in request_outputs:
                self.request_engine.process_request_output(request_output)
        

     def get_total_tokens(self):
         return self.master.get_number_of_total_tokens()
     
     async def generate_answer(
            self,
            request_id: str,
            **add_request_info
     ) -> ResponseOutput:

        # Preprocess the request.
        try:
           
            res_stream = await self.register_request(request_id, **add_request_info)
            
            async for request_output in res_stream:
                yield request_output
            
        except Exception as e:
            # If there is an exception, abort the request.
            self._abort(request_id)
            raise e
        
     async def _master_abort(self, request_ids):
        self.master.abort_request(request_ids)
