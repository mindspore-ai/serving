from typing import Dict, List, Optional
import time
import random
import logging

from serving_utils.entry import EntryData, EntryMetaData, EntryStatus
from .utils import Counter, ResponseOutput

from schedule.schedule import Schedule
from worker.worker import Worker
from config.serving_config import Baseconfig, AgentIP, AgentConfig, ModelName, transformer_tokenizer_path
from serving_utils.register import registers
from mindformers.mindformer_book import MindFormerBook
from mindformers import LlamaTokenizer
# from mindformers import AutoTokenizer
from transformers import AutoTokenizer

Eps = 30


def build_tokenizer(base_config):
    tokenizer = None
    if base_config.tokenizer in MindFormerBook.get_tokenizer_support_list():
        logging.info('load tokenizer from mindformers')
        tokenizer = AutoTokenizer.from_pretrained(base_config.tokenzier)
    else:
        if base_config.tokenizer == 'LlamaTokenizer':
            # tokenizer = LlamaTokenizer(base_config.tokenizer_path)
            tokenizer = AutoTokenizer.from_pretrained(transformer_tokenizer_path)
            print(f'tokenizer special tokens is {tokenizer.all_special_tokens}')
            return tokenizer
        # logging.info('load custom tokenizer')
        tokenizer = registers.TOKENIZER.get_obj_map()[base_config.tokenizer](base_config.tokenizer_path)
    return tokenizer


class Master:
    def __init__(self,
                 model_config=Baseconfig):
        self.model_config = model_config
        self.tokenizer = None
        self.counter = Counter()
        self.worker = Worker(AgentConfig.AgentPorts, AgentIP, ModelName)
        self.scheduler = Schedule(dyn_batch=model_config.dyn_batch_size,
                                  batch_size=model_config.batch_size,
                                  batch_waiting_time=model_config.batch_waiting_time,
                                  decode_batch_waiting_time=model_config.decode_batch_waiting_time,
                                  batching_strategy=model_config.batching_strategy,
                                  max_input_len=model_config.seq_length[-1] - Eps)

        self.is_running = False
        self._init_workers()
        self._counter_of_token = 0
        self._init_tokenizer()
        self.decode_cache = {}

    def _init_tokenizer(self):
        self.tokenizer = build_tokenizer(self.model_config)
        if self.tokenizer is None:
            logging.error('load tokenizer failed!')
        print(f'self.tokenizer is {self.tokenizer}')

    def _init_workers(self):
        self.worker._init_worker()

    def _schedule(self) -> List[EntryMetaData]:
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
        # prompt
        if index_list is not None:
            return [self._llama_detokenizer_function(index_list[0], entry_metadata_list, skip_special_tokens)]
        # decode
        str_outputs = []
        for index, output in enumerate(outputs):
            str_outputs.append(self._llama_detokenizer_function(index, entry_metadata_list, skip_special_tokens))
        return str_outputs

    def _postprocess(self,
                     outputs: List[int],
                     entry_metadata_list: List[EntryMetaData],
                     freq_list=[],
                     index_list: List[int] = None,
                     skip_inference=False) -> List[ResponseOutput]:

        end_token = self.model_config.end_token  # for debug
        self.scheduler.upate_entries_after_one_step(outputs, end_token, index_list)

        str_outputs = [''] * len(outputs)

        if self.model_config.tokenizer == 'LlamaTokenizer' and outputs[0] != -1:
            # str_outputs = self._llama_detokenizer(outputs)
            str_outputs = self._llama_detokenizer_v2(outputs, entry_metadata_list,
                                                     index_list, skip_special_tokens=True)

        elif self.model_config.tokenizer == 'InternLMTokenizer' and outputs[0] != -1:
            str_outputs = self._detokenizer(outputs)

        self._counter_of_token += len(outputs)
        logging.debug("current total token numbers is {}".format(self._counter_of_token))
        # generating output
        results: List[ResponseOutput] = []

        for index, output in enumerate(outputs):
            # prompt result, len(outputs) = 1
            if index_list is not None:
                if entry_metadata_list[index_list[0]].entry_data.status == EntryStatus.PADDING_INVAILED:
                    logging.debug(f'generate a invalid token, index in batch is {index}')
                    continue
                if skip_inference:
                    logging.debug(f'input out of range, index in batch is {index}')
                    results.append(ResponseOutput.generate_result(output,
                                                                  entry_metadata_list[index_list[0]],
                                                                  str_outputs[0],
                                                                  end_token, reason='Error202: prompt out of range'))
                    return results
                results.append(ResponseOutput.generate_result(output,
                                                              entry_metadata_list[index_list[0]],
                                                              str_outputs[0],
                                                              end_token))
            # encode result
            else:
                if entry_metadata_list[index].entry_data.status == EntryStatus.PADDING_INVAILED:
                    logging.debug(f'generate a invalid token, index in batch is {index}')
                    continue
                results.append(ResponseOutput.generate_result(output,
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
        if self.model_config.tokenizer == 'LlamaTokenizer':
            prompt_token_ids = self.tokenizer.encode(prompt)
        elif self.model_config.tokenizer == 'InternLMTokenizer':
            prompt_token_ids = self.tokenizer(prompt)['input_ids'][1:]
        else:
            print('incorrect Tokenizer name')
            logging.debug('incorrect Tokenizer name')
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

        entry_meta_data = EntryMetaData(request_id=request_id,
                                        is_prompt=True,
                                        entry_data=entry_data,
                                        entry_id=entry_id,
                                        prompt=prompt)

        logging.debug("add request to schedule queue {}".format(entry_meta_data.request_id))
        self.scheduler.add_entrys(entry_meta_data)

    def step(self) -> List[ResponseOutput]:
        # do inference
        batch_time = time.time()
        entry_metadata_list = self._schedule()
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
        entries_metadata_list, current_batch_size = self._schedule()
        valid_entry_len = 0
        for metadata in entries_metadata_list:
            logging.debug("entry_data status after schedule is {}".format(metadata.entry_data.get_status()))
            if metadata.entry_data.get_status() == EntryStatus.RUNNING or \
                    metadata.entry_data.get_status() == EntryStatus.INPUT_OUTOFRANGE:
                valid_entry_len += 1
        if valid_entry_len == 0:
            return

        logging.debug(f'valid entry_data is {valid_entry_len}')
        output = await self._run_workers_async(current_batch_size, entry_metadata_list=entries_metadata_list,
                                               model_config=self.model_config)
        post_process_time = time.time()
        # results = self._postprocess(output, entry_metadata_list=input_entry_metadata_list)
        logging.info('post_process_time time is {}'.format((time.time() - post_process_time) * 1000))
        return output

    async def _run_workers_async(self, current_batch_size, entry_metadata_list, model_config):
        e_t_e_time = time.time()
        input_entry_metadata_list = entry_metadata_list
        index_list = None
        # for item in entries_metadata_list:
        for index, item in enumerate(entry_metadata_list):
            if item.is_prompt:
                input_entry_metadata_list = [item]
                # optimize prefill multi-batch later
                index_list = [index]
                if item.entry_data.status == EntryStatus.INPUT_OUTOFRANGE:
                    return self._postprocess([111], entry_metadata_list=entry_metadata_list, index_list=index_list,
                                             skip_inference=True)
                else:
                    break
        logging.debug('len of input entry_metadata_list is {}'.format(len(input_entry_metadata_list)))
        # valid prompt add to batching list
        if len(input_entry_metadata_list) == 1 and \
                input_entry_metadata_list[0].entry_data.get_status() != EntryStatus.PADDING_INVAILED:
            output = self.worker.predict(current_batch_size, entry_metadata_list=input_entry_metadata_list, config=model_config)
        # invalid prompt add to batching list
        elif len(input_entry_metadata_list) == 1 and \
                input_entry_metadata_list[0].entry_data.get_status() == EntryStatus.PADDING_INVAILED:
            output = self.worker.predict(current_batch_size, entry_metadata_list=input_entry_metadata_list, config=model_config)
        # decode
        else:
            output = self.worker.predict(current_batch_size, entry_metadata_list=input_entry_metadata_list, config=model_config)
        result = self._postprocess(output, entry_metadata_list=entry_metadata_list, index_list=index_list)
        logging.info('e-to-e time is {}'.format((time.time() - e_t_e_time) * 1000))
        return result

    async def _mock_run_workers_async(self, batch_size: int):
        outputs = []
        for i in range(batch_size):
            output = random.randint(0, 32000)
            outputs.append(output)
        return outputs

    def stop(self):
        return self.worker.stop()
