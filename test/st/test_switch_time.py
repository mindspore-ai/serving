import time
import numpy as np
import mindspore_lite as mslite
from config.serving_config import Baseconfig, AgentConfig, prefill_model_path
from config.serving_config import device as device_id

def load_model(model_path, config_file, rank_id, device_id):
    context = mslite.Context()
    print('device_id: ', device_id)
    print('rank_id: ', rank_id)
    context.ascend.device_id = device_id
    context.ascend.rank_id = rank_id
    context.ascend.provider = "ge"
    context.target = ["Ascend"]

    # rank_table_file放在config_file中
    model = mslite.Model()

    print('load prefill model ...')
    print("prefill model path:", model_path, "config_file", config_file)
    start_time = time.time()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, context, config_file)
    print("model loaded, time:", (time.time() - start_time) * 1000, "ms")
    return model

def get_baichuan2_prefill_inputs(seq_length_index):
    seq_length = Baseconfig.seq_length[seq_length_index]
    prev_seq_length = 0 if seq_length_index == 0 else Baseconfig.seq_length[seq_length_index - 1]
    
    begin_slot_id = 128
    null_slot_id = 0

    valid_length = np.random.randint(prev_seq_length + 1, seq_length, dtype=np.int32)
    batch_valid_length = np.array([valid_length], dtype=np.int32)
    current_index = batch_valid_length - 1

    input_ids = np.random.randint(0, Baseconfig.vocab_size, size=valid_length, dtype=np.int32)
    input_ids = np.pad(input_ids, (0, seq_length - valid_length), 'constant', constant_values=0)
    input_ids = input_ids.reshape(1, seq_length)

    init_reset = np.array([False], dtype=np.bool_)
    slot_mapping = np.array(np.array(list(range(begin_slot_id, begin_slot_id + valid_length))), dtype=np.int32)
    slot_mapping = np.pad(slot_mapping, (0, seq_length - valid_length), 'constant', constant_values=null_slot_id)

    batch_index = np.array([0], dtype=np.int32)

    inputs_list = [input_ids, current_index, init_reset, batch_valid_length, batch_index, slot_mapping]

    return inputs_list

def predict(model, prefill_inputs_list):
    prefill_lite_inputs = [mslite.Tensor(item) for item in prefill_inputs_list]
    start_time = time.time()
    model.predict(prefill_lite_inputs)
    print("model predict time is:", (time.time() - start_time) * 1000, "ms")

if __name__ == "__main__":
    model = load_model(prefill_model_path[0], AgentConfig.ctx_setting, 0, device_id)
    # warmup
    seq_length_index_list = [
        2, 2, # 不换挡
        1, # 换挡第一次
        2, 1, # 尝试同样换挡时间
        0, 1, 2, 1, 0, # 连续档位切换
        2, 0, 2, 1, 0 # 非连续档位切换
        ]
    for i, index in enumerate(seq_length_index_list):
        prev_seq_length = Baseconfig.seq_length[seq_length_index_list[i-1]] if i > 0 else None
        print("step:", i, "test seq len: (", prev_seq_length, "-->)", Baseconfig.seq_length[index])
        inputs = get_baichuan2_prefill_inputs(index)
        for j, item in enumerate(inputs):
            print("input index:", j)
            print("value:", item)
            print("dtype:", item.dtype)
            print("shape:", item.shape)
        predict(model, inputs)
