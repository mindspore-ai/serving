from typing import Dict, List, Optional
import time
import random
import logging

from lib.entry import EntryData, EntryMetaData, EntryStatus
from .utils import Counter, ResponseOutput

from schedule.schedule import Schedule
from worker.worker import Worker
from config.serving_config import Baseconfig, AgentIP, AgentConfig, ModelName
from config.serving_config import get_token


class Master:
    def __init__(self,
                 model_config=Baseconfig):
        self.model_config = model_config
        self.tokenizer = get_token()
        self.counter = Counter()
        self.worker = Worker(AgentConfig.AgentPorts, AgentIP, ModelName)
        self.scheduler = Schedule(batch_size=model_config.batch_size, max_len=model_config.max_generate_length)    # requeset pool & get a batch for inference, such as continous batch, static batch etc.
        self.is_running = False
        self._init_workers()
        self._counter_of_token = 0

    def _init_workers(self):
        self.worker._init_worker()

    def _schedule(self) -> List[EntryMetaData]:
        return self.scheduler.schedule()

    def get_number_of_total_tokens(self):
        return self._counter_of_token
        
    def _detokenizer(self, outputs: List[int]) -> List[str]:
        str_outputs = []
        for output in outputs:
            output_ = self.tokenizer._convert_id_to_token(output)
            str_outputs.append(output_)
        return str_outputs

    def _postprocess(self,  
                     outputs: List[int],
                     entry_metadata_list: List[EntryMetaData], 
                     freq_list=[]) -> List[ResponseOutput]:
        # detokenizer, convert token(int) into text(str)
        end_token = 2   # for debug
        str_outputs = self._detokenizer(outputs)
        self._counter_of_token += len(outputs)
        logging.info("current total token numbers is {}".format(self._counter_of_token))
        self.scheduler.upate_entries_after_one_step(outputs, end_token)
        # generating output
        results: List[ResponseOutput] = []
        
        for index, output in enumerate(outputs):
            if entry_metadata_list[index].entry_data.status == EntryStatus.PADDING_INVAILED:
                continue
            results.append(ResponseOutput.generate_result(output, entry_metadata_list[index], str_outputs[index], end_token))
        return results

    def abort_request(self, request_id) -> None:
        
        self.scheduler.abort_entry(request_id)

    def add_requests_to_schedule_pool(self,
                                      request_id: str,
                                      prompt: Optional[str],
                                      do_sample,
                                      top_k,
                                      top_p,
                                      temperature,
                                      repetition_penalty,
                                      max_token_len
                                      ):
        time_tokenizer = time.time()
        prompt_token_ids = self.tokenizer.encode(prompt)
        logging.info('tokenizer time is {}'.format((time.time() - time_tokenizer) * 1000))
            
        # Create the sequences.
        entry_id = next(self.counter)
        entry_data = EntryData(prompt_tokens=prompt_token_ids,
                               max_token_len=max_token_len,
                               do_sample=do_sample,
                               tok_k=top_k,
                               top_p=top_p,
                               temperature=temperature,
                               repetition_penalty=repetition_penalty)
        entry_meta_data = EntryMetaData(request_id=request_id,
                                        is_prompt=True,
                                        entry_data=entry_data,
                                        entry_id=entry_id,
                                        prompt=prompt)
        logging.info("add request to schedule queue {}".format(entry_meta_data.request_id))
        self.scheduler.add_entrys(entry_meta_data)


    def step(self) -> List[ResponseOutput]:
        # do inference
        batch_time = time.time()
        entry_metadata_list, _= self._schedule()
        # Execute the model.
        # output: model infer out(token): 
        # output is batch_size * n_src_vocab
        output = self._mock_run_workers_async(entry_metadata_list=entry_metadata_list, model_config=self.model_config)
        return self._postprocess(output, entry_metadata_list)
    
    def _mock_run_workers_async(self, batch_size: int):
        outputs = []
        for i in range(batch_size):
            output = random.randint(0, 32000)
            outputs.append(output)
        time.sleep(0.15)
        return outputs
    

class AsyncMaster(Master):
    async def step_async(self) -> List[ResponseOutput]:
        batch_time = time.time()
        entries_metadata_list = self._schedule()
        if len(entries_metadata_list) == 0:
            return
        run_worker_time = time.time()
        output = await self._run_workers_async(entry_metadata_list=entries_metadata_list, model_config=self.model_config)
        post_process_time = time.time()
        results = self._postprocess(output, entry_metadata_list=entries_metadata_list)
        logging.info('e-to-e time is {}'.format((time.time() - batch_time) * 1000))
        logging.info('post_process_time time is {}'.format((time.time() - post_process_time) * 1000))
        return results

    async def _run_workers_async(self, entry_metadata_list, model_config):
        tim_ = time.time()
        return self.worker.predict(entry_metadata_list=entry_metadata_list, config=model_config)
    
    async def _mock_run_workers_async(self, batch_size: int):
        outputs = []
        for i in range(batch_size):
            output = random.randint(0, 32000)
            outputs.append(output)
        return outputs
