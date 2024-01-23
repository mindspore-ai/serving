import asyncio
import logging
import subprocess

from mindspore_serving.master.master import AsyncMaster
from mindspore_serving.master.response_async_queue import AsyncResultsOfOneRequest
from mindspore_serving.master.utils import ResponseOutput, ModelInfo
from mindspore_serving.master.request_resister_engine import RequestEngine
from mindspore_serving.config.config import ServingConfig


# from mindspore_serving.serving_utils.register import import_all_modules_for_register

# import_all_modules_for_register()


class LLMServer:
    """
       request_queue(FIFO): add request into a async queue, and monitor request status(is_finished),
                            mapping inference result of each iteration to corresponding request
                            result queue(used in stream return).
       master: Continuously getting unfinished request from request_queue, conducting batch strategy,
               and doing one step inference using ms-lite, after get result of one iteration, client
               get stream inference result from request_queue, and update to request_queue.
    """

    def __init__(self, config: ServingConfig):
        self.request_engine = RequestEngine()
        self.background_loop = None
        self.master = AsyncMaster(config)
        self.status = 0
        self.config = config

    @property
    def is_running(self) -> bool:
        return self.background_loop is not None

    async def run_loop(self):
        while self.status:
            await self.step()
            await asyncio.sleep(0)

    def start_background_loop(self) -> None:
        # todo
        self.status = 1
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self.background_loop = asyncio.get_event_loop().create_task(self.run_loop())

    async def register_request(self,
                               request_id: str,
                               **add_request_info) -> AsyncResultsOfOneRequest:
        logging.debug("background loop {}".format(self.background_loop))
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
        if self.status == 0:
            return
        new_requests, finished_requests = self.request_engine.get_requests_from_register_pool()
        for new_request in new_requests:
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

    def get_bs_current(self):
        return self.master.get_current_batch()

    def get_queue_current(self):
        return self.master.get_current_requestes_nums()

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

    def stop(self):
        # 1. stop background
        self.status = 0
        self.master.stop()

    def get_dockerId(self):
        p = subprocess.Popen("cat /proc/self/cgroup | grep /docker | head -1 | cut -d/ -f3", shell=True,
                             stdout=subprocess.PIPE)
        out = p.stdout.read()
        id = str(out, 'utf-8')
        return id

    def get_serverd_model_info(
            self
    ) -> ModelInfo:
        max_seq_length = int(self.config.model_config.seq_length[-1])
        max_decode_batch_size = int(self.config.model_config.decode_batch_size[-1])
        docker_id = self.get_dockerId()
        serverd_model_info = ModelInfo(docker_label=docker_id,
                                       max_batch_total_tokens=max_seq_length * max_decode_batch_size,
                                       max_concurrent_requests=self.master.get_current_requestes_nums(),
                                       max_input_length=max_seq_length, max_total_tokens=max_decode_batch_size,
                                       model_dtype=self.config.model_config.model_dtype,
                                       model_id=self.config.model_config.model_name)
        return serverd_model_info
