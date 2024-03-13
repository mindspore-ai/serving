from tabnanny import check
from typing import List, Optional, Tuple
import copy
import time
import random
import logging

from mindspore_serving.serving_utils.entry import EntryData, EntryMetaData, EntryStatus
from .utils import Counter, ResponseOutput

from mindspore_serving.schedule.schedule import Schedule
from mindspore_serving.worker.worker import Worker
from mindspore_serving.config.config import ServingConfig
from mindspore_serving.models.build_tokenizer import build_tokenizer
from mindspore_serving.schedule.cache_engine import ServingBlockMemPool
from mindspore_serving.serving_utils.constant import *
from mindformers.mindformer_book import MindFormerBook

Eps = 30


class Master:
    def __init__(self,
                 config: ServingConfig):
        self.config = config
        self.tokenizer = None
        self.counter = Counter()
        self.worker = Worker(config)
        self.scheduler = Schedule(config)

        self.is_running = False
        self._init_workers()
        self._counter_of_token = 0
        self._init_tokenizer()
        self.decode_cache = {}
        if self.config.model_config.page_attention:
            self._init_mem_pool()  # PA

    # PA

    def _init_mem_pool(self):
        ServingBlockMemPool.init(self.config.pa_config.num_blocks, self.config.pa_config.block_size)

    def _init_tokenizer(self):
        self.tokenizer = build_tokenizer(self.config)
        if self.tokenizer is None:
            logging.error('load tokenizer failed!')
        logging.debug(f'self.tokenizer is {self.tokenizer}')

    def _init_workers(self):
        self.worker._init_worker()

    def _schedule(self) -> Tuple[List[EntryMetaData], int]:
        return self.scheduler.schedule()

    def get_number_of_total_tokens(self):
        return self._counter_of_token

    def _detokenizer(self, tokens: List[int]) -> List[str]:
        """
           tokens is results of post-sampling module.
           output: texts list of batch
        """
        texts = []
        for token in tokens:
            token_input = [token]
            text = self.tokenizer.decode(token_input, skip_special_tokens=True)
            logging.debug(f'tokenizer decode result is {text}, token id is {token}')
            texts.append(text)
        return texts

    def _llama_detokenizer(self, outputs):
        str_outputs = []
        batch_size = len(outputs)
        before_batch_size = len(self.decode_cache.keys())
        if batch_size > before_batch_size:
            for i in range(before_batch_size, batch_size):
                self.decode_cache[i] = []
        else:
            while len(self.decode_cache.keys()) > batch_size:
                self.decode_cache.popitem()
        for i in range(batch_size):
            self.decode_cache[i].append(outputs[i])
            new_text = self.tokenizer.decode(self.decode_cache[i], skip_special_tokens=True)
            if not new_text.endswith("�"):
                begin_token = self.tokenizer._convert_id_to_token(self.decode_cache[i][0])
                if begin_token == '<0x0A>':
                    begin_token = '\n'
                elif '\u2581' in begin_token:
                    begin_token = ' '
                else:
                    begin_token = ''

                str_outputs.append(begin_token + new_text)
                self.decode_cache[i] = []
            else:
                str_outputs.append('')
        return str_outputs

    def _llama_detokenizer_function(self, index, entry_metadata_list, skip_special_tokens=True):

        prefix_index = entry_metadata_list[index].get_entry_data().prefix_index
        read_index = entry_metadata_list[index].get_entry_data().read_index
        if entry_metadata_list[index].get_entry_data().get_status() != EntryStatus.RUNNING:
            return ""
        all_outputs_ids = entry_metadata_list[index].get_entry_data().get_output_token()

        prefix_text = self.tokenizer.decode(all_outputs_ids[prefix_index: read_index],
                                            skip_special_tokens=skip_special_tokens)

        new_text = self.tokenizer.decode(all_outputs_ids[prefix_index:], skip_special_tokens=skip_special_tokens)

        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            new_text = new_text[len(prefix_text):]
            entry_metadata_list[index].get_entry_data().prefix_index = read_index
            entry_metadata_list[index].get_entry_data().read_index = len(all_outputs_ids)
            return new_text
        else:
            return ""

    def _llama_detokenizer_v2(self,
                              outputs,
                              entry_metadata_list,
                              index_list=None,
                              skip_special_tokens=True):
        str_outputs = []
        # prompt
        if index_list is not None:
            for index in index_list:
                str_outputs.append(self._llama_detokenizer_function(index, entry_metadata_list, skip_special_tokens))
            return str_outputs
        # decode
        for index, output in enumerate(outputs):
            str_outputs.append(self._llama_detokenizer_function(index, entry_metadata_list, skip_special_tokens))
        return str_outputs

    def _check_error_code(self, output_token):
        error_code_list = [-1, -202, -203]
        if output_token in error_code_list:
            return True
        return False

    def _postprocess(self,
                     outputs: List[tuple],
                     entry_metadata_list: List[EntryMetaData],
                     index_list: List[int] = None,
                     skip_inference=False) -> List[ResponseOutput]:

        end_token = self.config.model_config.end_token  # for debug

        output_tokens = []
        output_logprob = []
        for output_tup in outputs:
            output_tokens.append(output_tup[0])
            output_logprob.append(output_tup[1])

        self.scheduler.upate_entries_after_one_step(output_tokens, end_token, index_list)
        str_outputs = [''] * len(output_tokens)
        if (self.config.model_config.model_name.startswith(
                'llama') or self.config.model_config.model_name == 'wizard_coder') and not self._check_error_code(
            output_tokens[0]):
            # str_outputs = self._llama_detokenizer(outputs)
            str_outputs = self._llama_detokenizer_v2(output_tokens, entry_metadata_list,
                                                     index_list, skip_special_tokens=True)

        elif self.config.model_config.model_name in (
                'internlm_7b', 'baichuan2pa', 'gpt2') and not self._check_error_code(output_tokens[0]):
            str_outputs = self._detokenizer(output_tokens)
        self._counter_of_token += len(output_tokens)
        logging.debug("target is {}, str_outputs is {}".format(outputs, str_outputs))
        logging.debug("current total token numbers is {}".format(self._counter_of_token))
        # generating output
        results: List[ResponseOutput] = []
        # prompt result
        if index_list is not None:
            # idx: index_list and outputs data index, index: batch list index.
            for idx, index in enumerate(index_list):
                if entry_metadata_list[index].entry_data.status == EntryStatus.PADDING_INVAILED:
                    logging.debug(f'generate a invalid token, index in batch is {index}')
                    continue
                if output_tokens[0] == INPUT_OUT_OF_TOKEN[0]:
                    logging.debug(f'input out of range, index in batch is {index}')
                    results.append(ResponseOutput.generate_result(output_tokens[idx],
                                                                  0,
                                                                  entry_metadata_list[index],
                                                                  str_outputs[idx],
                                                                  end_token, reason='Error202: prompt out of range'))
                    return results

                if output_tokens[0] == INPUT_EMPTY_TOKEN[0]:
                    logging.debug(f'prompt token empty, index in batch is {index}')
                    results.append(ResponseOutput.generate_result(output_tokens[idx],
                                                                  0,
                                                                  entry_metadata_list[index],
                                                                  str_outputs[idx],
                                                                  end_token, reason='Error203: prompt token empty'))
                    return results

                results.append(ResponseOutput.generate_result(output_tokens[idx],
                                                              output_logprob[idx],
                                                              entry_metadata_list[index],
                                                              str_outputs[idx],
                                                              end_token))
        # encode result
        else:
            for index, output_token in enumerate(output_tokens):
                output_token_logprob = output_logprob[index]
                if entry_metadata_list[index].entry_data.status == EntryStatus.PADDING_INVAILED:
                    logging.debug(f'generate a invalid token, index in batch is {index}')
                    continue
                results.append(ResponseOutput.generate_result(output_token,
                                                              output_token_logprob,
                                                              entry_metadata_list[index],
                                                              str_outputs[index],
                                                              end_token))
        return results

    def get_current_batch(self):
        return self.scheduler.get_dyn_batch()

    def get_current_requestes_nums(self):
        return self.scheduler.get_queue_len()

    def abort_request(self,
                      request_id: str) -> None:
        self.scheduler.abort_entry(request_id)

    def add_requests_to_schedule_pool(self,
                                      request_id: str,
                                      prompt: Optional[str],
                                      do_sample,
                                      top_k,
                                      top_p,
                                      temperature,
                                      repetition_penalty,
                                      max_token_len):
        time_tokenizer = time.time()
        prompt_token_ids = None
        logging.debug("request id add_requests_to_schedule_pool {}".format(request_id))
        # 加入baichuan
        if self.config.model_config.model_name in (
                'baichuan2pa', 'wizard_coder') or self.config.model_config.model_name.startswith('llama'):
            prompt_token_ids = self.tokenizer.encode(prompt)
        elif self.config.model_config.model_name == 'internlm_7b':
            prompt_token_ids = self.tokenizer(prompt)['input_ids'][1:]
        elif self.config.model_config.model_name in MindFormerBook.get_tokenizer_support_list():
            prompt_token_ids = self.tokenizer(prompt)['input_ids']
        else:
            print('incorrect model_name')
            logging.debug('incorrect model_name')

        logging.info('tokenizer result prompt_token_ids is {}'.format(prompt_token_ids))
        logging.info('tokenizer time is {}'.format((time.time() - time_tokenizer) * 1000))

        # if prompt_token_ids is not None and
        # Create the sequences.
        entry_id = next(self.counter)
        entry_data = EntryData(prompt_tokens=prompt_token_ids,
                               max_token_len=max_token_len,
                               do_sample=do_sample,
                               tok_k=top_k,
                               top_p=top_p,
                               temperature=temperature,
                               repetition_penalty=repetition_penalty)
        block_size = 0
        if self.config.model_config.page_attention:
            block_size = self.config.pa_config.block_size
        entry_meta_data = EntryMetaData(page_attention=self.config.model_config.page_attention,
                                        request_id=request_id,
                                        is_prompt=True,
                                        entry_data=entry_data,
                                        entry_id=entry_id,
                                        prompt=prompt,
                                        block_size=block_size)

        logging.debug("add request to schedule queue {}".format(entry_meta_data.request_id))
        self.scheduler.add_entrys(entry_meta_data)

    def step(self) -> List[ResponseOutput]:
        # do inference
        entry_metadata_list, batch_size = self._schedule()
        # Execute the model.
        # output: model infer out(token):
        # output is batch_size * n_src_vocab
        output = self._mock_run_workers_async(batch_size)
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
        entries_metadata_list, current_batch_size = self._schedule()
        valid_entry_len = 0
        for metadata in entries_metadata_list:
            if metadata.entry_data.get_status() == EntryStatus.RUNNING or \
                    metadata.entry_data.get_status() == EntryStatus.INPUT_OUTOFRANGE:
                valid_entry_len += 1
        if valid_entry_len == 0:
            return None

        output = await self._run_workers_async(current_batch_size, entry_metadata_list=entries_metadata_list)
        return output

    def _get_prompt_batch_list(self, entry_metadata_list):
        # PA取一个data进行prefill
        if self.config.model_config.page_attention:
            return self._check_prompt_predict_data_pa(entry_metadata_list)
        input_entry_metadata_list, index_list = self._check_prompt_predict_data(entry_metadata_list)
        prompt_data_count = len(input_entry_metadata_list)

        if prompt_data_count == 0:
            return input_entry_metadata_list, index_list
        logging.debug("_get_prompt_batch_list prompt index_list {}, input_entry_metadata_list {}"
                      .format(index_list, input_entry_metadata_list))

        prefill_batch_size_list = self.config.model_config.prefill_batch_size
        if prefill_batch_size_list is None or len(prefill_batch_size_list) == 0:
            return [input_entry_metadata_list[0]], [index_list[0]]
        else:  # pure dyn
            dyn_bach_size = prefill_batch_size_list[0]
            if prompt_data_count > dyn_bach_size:
                input_entry_metadata_list = input_entry_metadata_list[:dyn_bach_size]
                index_list = index_list[:dyn_bach_size]

        return input_entry_metadata_list, index_list

    @staticmethod
    def get_last_prompt_entry(entry_metadata_list):
        for i in range(len(entry_metadata_list) - 1, -1, -1):
            entry_meta_data = entry_metadata_list[i]
            if entry_meta_data.is_prompt:
                return entry_meta_data

    @staticmethod
    def _get_prefill_padding_entry(index, entry_meta_data):
        copy_entry_meta_data = copy.deepcopy(entry_meta_data)
        copy_entry_meta_data.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)
        copy_entry_meta_data.get_entry_data().set_decode_index(index)
        logging.debug(f'add invalid request into prefill batch list, batch size index is {index}')
        return copy_entry_meta_data

    @staticmethod
    def _check_prompt_out_of_range_index_list(entry_metadata_list):
        """check prompt out of range index list"""
        out_of_range_index_list = []
        # for item in entries_metadata_list:
        for index, item in enumerate(entry_metadata_list):
            if not item.is_prompt or item.entry_data.status != EntryStatus.INPUT_OUTOFRANGE:
                continue

            out_of_range_index_list.append(index)
        return out_of_range_index_list

    @staticmethod
    def _check_prompt_predict_data_pa(entry_metadata_list):
        input_entry_metadata_list = []
        index_list = []
        for index, item in enumerate(entry_metadata_list):
            if not item.is_prompt or item.entry_data.status == EntryStatus.INPUT_OUTOFRANGE:
                continue
            input_entry_metadata_list = [item]
            index_list = [index]
            break
        return input_entry_metadata_list, index_list

    @staticmethod
    def _check_prompt_predict_data(entry_metadata_list):
        input_entry_metadata_list = []
        index_list = []
        for index, item in enumerate(entry_metadata_list):
            if not item.is_prompt or item.entry_data.status == EntryStatus.INPUT_OUTOFRANGE:
                continue

            input_entry_metadata_list.append(item)
            index_list.append(index)
        return input_entry_metadata_list, index_list

    @staticmethod
    def _check_prompt_token_empty(entry_metadata_list, pad_token_id):
        empty_list = []
        for index, item in enumerate(entry_metadata_list):
            if item.get_entry_data().get_prompt_token() == None or item.get_entry_data().get_prompt_len() == 0:
                item.get_entry_data().set_status(EntryStatus.EMPTY_PROMPT_TOKEN)
                empty_list.append(index)
            if pad_token_id in item.get_entry_data().get_prompt_token():
                item.get_entry_data().set_status(EntryStatus.EMPTY_PROMPT_TOKEN)
                empty_list.append(index)
        return empty_list

    async def _run_workers_async(self, current_batch_size, entry_metadata_list):
        e_t_e_time = time.time()

        prompt_token_empty_list = self._check_prompt_token_empty(entry_metadata_list,
                                                                 self.config.model_config.pad_token_id)
        logging.debug("prompt token empty list index_list {}".format(prompt_token_empty_list))
        if len(prompt_token_empty_list) > 0:
            return self._postprocess([INPUT_EMPTY_TOKEN], entry_metadata_list=entry_metadata_list,
                                     index_list=prompt_token_empty_list,
                                     skip_inference=True)

        # check prefill out of range data
        out_of_range_index_list = self._check_prompt_out_of_range_index_list(entry_metadata_list)
        logging.debug("out of range prompt index_list {}".format(out_of_range_index_list))
        if len(out_of_range_index_list) > 0:
            return self._postprocess([INPUT_OUT_OF_TOKEN], entry_metadata_list=entry_metadata_list,
                                     index_list=out_of_range_index_list,
                                     skip_inference=True)

        # filter prompt data batch list
        input_entry_metadata_list, index_list = self._get_prompt_batch_list(entry_metadata_list)
        logging.debug("_get_prompt_batch_list prompt index_list {}, input_entry_metadata_list {}"
                      .format(index_list, input_entry_metadata_list))
        # prefill predict
        if len(input_entry_metadata_list) > 0:
            logging.debug('prefill len of input entry_metadata_list is {}'.format(len(input_entry_metadata_list)))
            # predict
            output = self.worker.predict(current_batch_size, entry_metadata_list=input_entry_metadata_list)
        else:  # decode predict
            input_entry_metadata_list = entry_metadata_list
            index_list = None
            logging.debug('decode len of input entry_metadata_list is {}'.format(len(input_entry_metadata_list)))
            output = self.worker.predict(current_batch_size, entry_metadata_list=input_entry_metadata_list)

        post_process_time = time.time()
        result = self._postprocess(output, entry_metadata_list=entry_metadata_list, index_list=index_list)
        logging.info('post_process_time time is {}'.format((time.time() - post_process_time) * 1000))
        logging.info('e-to-e time is {}'.format((time.time() - e_t_e_time) * 1000))
        return result

    async def _mock_run_workers_async(self, batch_size: int):
        outputs = []
        for i in range(batch_size):
            output = random.randint(0, 32000)
            outputs.append(output)
        return outputs

    def stop(self):
        self.worker.stop()
