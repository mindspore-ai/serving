import time
from typing import Dict, List
import logging

from lib.entry import EntryMetaData, EntryStatus
from queue import Queue
import copy


class Schedule:
    """static batch strategy"""
    def __init__(self, batch_size: int=4, max_len: int=128):
        self.waiting_request_queue: Queue[EntryMetaData] = Queue()
        self.running_request_list: List[EntryMetaData] = []
        self.count_of_invalid_sample = 0
        self.batch_size = batch_size

    def add_entrys(self, entry_meta_data: EntryMetaData):
        entry_meta_data.get_entry_data().set_status(EntryStatus.WAITING)
        self.waiting_request_queue.put_nowait(entry_meta_data)

    def _padding_batch_size(self):
        while len(self.running_request_list) < self.batch_size:
            entry_meta_data = copy.deepcopy(self.running_request_list[-1])
            entry_meta_data.entry_data.set_status(EntryStatus.PADDING_INVAILED)
            self.running_request_list.append(entry_meta_data)

    def _over_all_complete_entry(self):
        for index, _ in enumerate(self.running_request_list):
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            
    def _get_next_batch(self):
        self.running_request_list.clear()
        count = 0

        if self.waiting_request_queue.empty():
            return
        
        while not self.waiting_request_queue.empty():
            if count >= self.batch_size:
                break
            data = self.waiting_request_queue.get_nowait()
            data.get_entry_data().set_status(EntryStatus.RUNNING)        
            self.running_request_list.append(data)
            count += 1
        
        if len(self.running_request_list) < self.batch_size:
            self._padding_batch_size()

    def _all_samples_in_batch_is_over(self) -> bool:
        res = True
        for _, data in enumerate(self.running_request_list):
            if data.get_entry_data().get_status() == EntryStatus.RUNNING:
                res = False
        return res            

    def _static_batch(self):
        if self._all_samples_in_batch_is_over() or len(self.running_request_list) == 0:
            self._get_next_batch()
        # updata status after one inference step
        for index, data in enumerate(self.running_request_list):
            data_status =  data.get_entry_data().get_status()
            if data_status != EntryStatus.RUNNING and data_status != EntryStatus.WAITING_BATCH:
                self.running_request_list[index].get_entry_data().set_status(EntryStatus.WAITING_BATCH)
                self.count_of_invalid_sample += 1
            # if all samples in batch is invalid status, a static batch is over
            if self.count_of_invalid_sample == self.batch_size:
                self._get_next_batch()

    def schedule(self) -> List[EntryMetaData]:
        self._static_batch()
        return self.running_request_list
    
    def upate_entries_after_one_step(self, outputs: List[int], eos_id: int):
        """update status after ever iteration"""
        for index, token in enumerate(outputs):
            if self.running_request_list[index].entry_data.status == EntryStatus.PADDING_INVAILED:
                continue
            self.running_request_list[index].get_entry_data().updata_output_tokens(token)
            if token == eos_id:
                logging.info("a request finished, token equal to {}".format(token))
                self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            entry_data = self.running_request_list[index].get_entry_data()
            if entry_data.max_token_len < entry_data.get_output_len():
                self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_LENGTH_CAPPED)
            if self.running_request_list[index].entry_data.get_output_len() != 0:
                self.running_request_list[index].is_prompt = False

    def abort_entry(self, request_id: str):
        for index, data in enumerate(self.running_request_list):
            if data.request_id == request_id:
                self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)