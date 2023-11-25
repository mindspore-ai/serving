import time
import uuid
from typing import Dict, List
import logging

from serving_utils.entry import EntryMetaData, EntryStatus, EntryData
from queue import Queue
import copy
from config.serving_config import Baseconfig
from serving_utils.constant import *

class Schedule:
    """static batch strategy"""

    def __init__(self,
                 dyn_batch=None,
                 batch_size: int = 4,
                 max_len: int = 128,
                 eos_token: int = 2,
                 batch_waiting_time: float = 0.3,
                 decode_batch_waiting_time: float = 0.06,
                 batching_strategy: str = 'static',
                 max_input_len: int = 8192):
        self.waiting_request_queue: Queue[EntryMetaData] = Queue()
        self.running_request_list: List[EntryMetaData] = []
        self.count_of_invalid_sample = 0
        self.batch_size = batch_size if dyn_batch is None or len(dyn_batch) <= 1 else 0
        self.eos_token = eos_token
        self.batch_waiting_time = batch_waiting_time
        self.decode_batch_waiting_time = decode_batch_waiting_time
        self.batching_strategy = batching_strategy
        self.max_input_len = max_input_len
        # batch中有效token的最大index, 初始化为-1
        self.max_valid_index = -1
        self.dyn_batch = dyn_batch

    def get_dyn_batch(self):
        return self.batch_size

    def get_queue_len(self):
        return self.waiting_request_queue.qsize()

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

    def _padding_request_into_batching_list(self, index):
        if self.waiting_request_queue.empty():
            time.sleep(self.batch_waiting_time / float(len(self.running_request_list)))
            if self.waiting_request_queue.empty():
                entry_meta_data = copy.deepcopy(self.running_request_list[-1])

                if entry_meta_data.entry_data.get_prompt_len() >= self.max_input_len:
                    entry_meta_data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
                else:
                    entry_meta_data.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)

                entry_meta_data.get_entry_data().set_decode_index(index)
                self.running_request_list.append(entry_meta_data)
                logging.info(f'waiting and add invalid request in batch init, batch size index is {index}')
            else:
                data = self.waiting_request_queue.get_nowait()
                if data.entry_data.get_prompt_len() >= self.max_input_len:
                    data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
                else:
                    data.get_entry_data().set_status(EntryStatus.RUNNING)

                data.get_entry_data().set_decode_index(index)
                self.running_request_list.append(data)
                logging.info(f'add new valid request in batch, batch size index is {index}')
        else:
            data = self.waiting_request_queue.get_nowait()
            logging.info('get_nowait2')

            if data.entry_data.get_prompt_len() >= self.max_input_len:
                data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
            else:
                data.get_entry_data().set_status(EntryStatus.RUNNING)

            data.get_entry_data().set_decode_index(index)
            self.running_request_list.append(data)
            logging.info(f'add new valid request in batch, batch size index is {index}')

    def _get_next_batch(self):
        self.running_request_list.clear()
        count = 0
        # no request in schedule queue, return
        if self.waiting_request_queue.empty():
            return
        # add request into batching list
        while not self.waiting_request_queue.empty():
            if count > self.batch_size:
                break
            data = self.waiting_request_queue.get_nowait()

            if data.entry_data.get_prompt_len() >= self.max_input_len:
                data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
            else:
                data.get_entry_data().set_status(EntryStatus.RUNNING)

            data.get_entry_data().set_decode_index(count)
            self.running_request_list.append(data)
            logging.info(f'add new valid request in batch, batch size index is {count}')
            count += 1
        # if batching list not full, add invalid padding request into batching list
        if len(self.running_request_list) < self.batch_size + 1:
            for index in range(len(self.running_request_list), self.batch_size):
                self._padding_request_into_batching_list(index)

    def _all_samples_in_batch_is_over(self) -> bool:
        res = True
        for _, data in enumerate(self.running_request_list):
            if data.get_entry_data().get_status() == EntryStatus.RUNNING:
                res = False
        return res

    def checkout_entry(self) -> List[bool]:
        """
          request in FINISHED_LENGTH_CAPPED, FINISHED_STOPPED, PADDING_INVAILED status can be cut out
        """
        checkout_list = []
        for index, data in enumerate(self.running_request_list):
            check_ = False
            # max_length, cut out finished request in batch
            if data.get_entry_data().get_status() == EntryStatus.FINISHED_LENGTH_CAPPED:
                check_ = True
            # eos, cut out finished request in batch
            elif data.get_entry_data().get_status() == EntryStatus.FINISHED_STOPPED:
                check_ = True
            elif data.get_entry_data().get_status() == EntryStatus.PADDING_INVAILED:
                check_ = True
            checkout_list.append(check_)
        return checkout_list

    def _padding_new_prompt_to_batch(self, index):
        # queue is empty, no new request in schedule queue
        if self.waiting_request_queue.empty():
            # waiting
            time.sleep(self.batch_waiting_time / float(len(self.running_request_list)))
            # no new request, continue finished valid decode
            if self.waiting_request_queue.empty():
                # logging.info('waiting and no new request, continue finished valid decode')
                return
            # new requestes in queue
            else:
                logging.info('add a new request into batching list')
                data = self.waiting_request_queue.get_nowait()
                logging.info('get_nowait3')
                if data.entry_data.get_prompt_len() >= self.max_input_len:
                    data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)

                else:
                    data.get_entry_data().set_status(EntryStatus.RUNNING)
                data.get_entry_data().set_decode_index(index)
                self.running_request_list[index] = data
                logging.info(f'add new valid request in batch, batch size index is {index}')
        else:
            logging.info('add a new request into batching list')
            data = self.waiting_request_queue.get_nowait()
            logging.info('get_nowait4')
            if data.entry_data.get_prompt_len() >= self.max_input_len:
                data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
            else:
                data.get_entry_data().set_status(EntryStatus.RUNNING)
            data.get_entry_data().set_decode_index(index)
            self.running_request_list[index] = data
            logging.info(f'add new valid request in batch, batch size index is {index}')

    def _update_status_after_one_itreation(self):
        self.count_of_invalid_sample = 0
        """checkout and update number of invalid request in batching list"""
        self.max_valid_index = -1
        for index, data in enumerate(self.running_request_list):
            data_status = data.get_entry_data().get_status()
            if data_status == EntryStatus.FINISHED_STOPPED or data_status == EntryStatus.FINISHED_LENGTH_CAPPED:
                self.count_of_invalid_sample += 1
            elif data_status == EntryStatus.RUNNING:
                self.max_valid_index = index

    def _determine_batch_size(self):
        self._update_status_after_one_itreation()
        bf_batch = self.batch_size
        queue_len = self.waiting_request_queue.qsize()
        bs_list_len = len(self.dyn_batch)
        # 1. 请求队列长度大于当前batch_size，扩容
        dyn_index = self.max_valid_index + 1
        if self.max_valid_index == -1 or queue_len > self.batch_size:
            # 获取最接近waiting list长度的batch档位
            dyn_index = queue_len
        # 2. 请求队列长度小于count_of_invalid_sample，根据max_valid_index动态到邻近档位
        elif queue_len < self.count_of_invalid_sample:
            # max_valid_index左侧有多少结束的token
            left_free_num = self.count_of_invalid_sample - (self.batch_size - self.max_valid_index - 1)
            if queue_len <= left_free_num:
                dyn_index = self.max_valid_index + 1
            else:
                # 请求队列中全部补齐会到哪个index
                dyn_index = queue_len - left_free_num + self.max_valid_index + 1
        else:
            dyn_index = self.max_valid_index + 1 + queue_len - self.count_of_invalid_sample
        bs_after_changing = self.batch_size
        if dyn_index <= 0:
            # 默认值
            bs_after_changing = self.dyn_batch[0]
        else:
            for i in range(1, bs_list_len):
                if dyn_index > self.dyn_batch[bs_list_len - i - 1]:
                    bs_after_changing = self.dyn_batch[bs_list_len - i]
                    break
        self.batch_size = bs_after_changing if bs_after_changing > 0 else self.dyn_batch[0]
        af_batch = self.batch_size
        if af_batch != bf_batch:
            logging.info(('----bs changed from  {} '.format(bf_batch)))
            logging.info(('----bs changed to  {} '.format(af_batch)))
            logging.info(('----dyn_index to  {} max_valid_index {}'.format(dyn_index, self.max_valid_index)))
        if bf_batch >= af_batch:
            self.running_request_list = self.running_request_list[:af_batch]
        else:
            bf_batch = 0 if self.max_valid_index == -1 else bf_batch
            for i in range(bf_batch, af_batch):
                entry_meta_data = EntryMetaData(request_id=PADDING_REQUEST_ID,
                                                is_prompt=True,
                                                entry_data=EntryData(prompt_tokens=[Baseconfig.end_token],
                                                                     max_token_len=5000),
                                                entry_id=-1,
                                                prompt=PADDING_PROMPT)
                entry_meta_data.get_entry_data().set_decode_index(i)
                entry_meta_data.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)
                self.running_request_list.append(entry_meta_data)
        # 3. 请求队列长度大于count_of_invalid_sample且小于当前batch_size，batch不变
            logging.info(('----padding running list  {} '.format(len(self.running_request_list))))

    def _continuous_batch(self):
        # init batch size when running_request_list is empty.
        if len(self.running_request_list) == 0:
            self._get_next_batch()
        # update invalid request number in batching list.
        self._update_status_after_one_itreation()
        if self.count_of_invalid_sample == self.batch_size:
            self._get_next_batch()
        # update status after one inference step
        else:
            checkout_list = self.checkout_entry()
            for index, data in enumerate(checkout_list):
                if data and index < self.batch_size:
                    # logging.info('have eos or max_length in batching list')
                    logging.info('----{}-th prefill request in batching padded to batch.'.format(index))
                    self._padding_new_prompt_to_batch(index)

    def _static_batch(self):
        if self._all_samples_in_batch_is_over() or len(self.running_request_list) == 0:
            self._get_next_batch()
        # updata status after one inference step
        self._update_status_after_one_itreation()
        # if all samples in batch is invalid status, a static batch is over
        if self.count_of_invalid_sample == self.batch_size:
            self._get_next_batch()

    def schedule(self) -> List[EntryMetaData]:
        if self.dyn_batch and len(self.dyn_batch) > 1:
            self._determine_batch_size()
        if self.batching_strategy == 'static':
            self._static_batch()
        elif self.batching_strategy == 'continuous':
            self._continuous_batch()
        else:
            raise ValueError("Invalid batching strategy!, please setting static or continuous")
        return self.running_request_list, self.batch_size

    def _finished_request(self, index, token, eos_id):
        # eos
        if token == eos_id:
            logging.info("a request finished, token equal to {}".format(token))
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return

        # max len
        entry_data = self.running_request_list[index].get_entry_data()
        if entry_data.max_token_len <= entry_data.get_output_len():
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_LENGTH_CAPPED)
            return

        # input outofrange
        if entry_data.status == EntryStatus.INPUT_OUTOFRANGE:
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return

    def upate_entries_after_one_step(self, outputs: List[int], eos_id: int, index_list: List[int] = None):
        """update status after ever iteration"""
        # optimize prefill multi-batch later
        if index_list is not None:
            for index in index_list:
                self.running_request_list[index].get_entry_data().updata_output_tokens(outputs[0])
                self.running_request_list[index].is_prompt = False
                # invalid prompt
                if self.running_request_list[index].get_entry_data().get_status() == EntryStatus.PADDING_INVAILED:
                    return
                # valid prompt
                else:
                    self._finished_request(index, outputs[0], eos_id)
        # decode
        else:
            for index, token in enumerate(outputs):
                # update new token to result list
                self.running_request_list[index].get_entry_data().updata_output_tokens(token)
                self._finished_request(index, token, eos_id)

    def abort_entry(self,
                    request_id: str):
        for index, data in enumerate(self.running_request_list):
            if data.request_id == request_id:
                self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
