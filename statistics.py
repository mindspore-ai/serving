import numpy as np
from numpy import *

e2e_fname = '/home/sc/gsp/LLM-Serving/1017_e2e_time.txt'
predict_fname = '/home/sc/gsp/LLM-Serving/1017_predict_time.txt'
npu_total_fname = '/home/sc/gsp/LLM-Serving/1017_npu_total_time.txt'

end_to_end = []
predict = []
npu_total = []

with open(e2e_fname, "r") as f:
    data = f.readlines()
    for line in data:
        end_to_end.append(float(line.split(' ')[-1]))

end_to_end.sort()
print("end_to_end time 50: ", end_to_end[int(len(end_to_end) / 2)])
print("end_to_end time 90: ", end_to_end[int(len(end_to_end) * 0.9)])
print("end_to_end time mean: ", np.mean(end_to_end))
print("end_to_end time max: ", max(end_to_end))
print("end_to_end time min: ", min(end_to_end))

with open(predict_fname, "r") as f:
    data = f.readlines()
    for line in data:
        predict.append(float(line.split(' ')[-1]))

predict.sort()
print("predict time 50: ", predict[int(len(predict) / 2)])
print("predict time 90: ", predict[int(len(predict) * 0.9)])
print("predict time mean: ", np.mean(predict))
print("predict time max: ", max(predict))
print("predict time min: ", min(predict))

with open(npu_total_fname, "r") as f:
    data = f.readlines()
    for line in data:
        npu_total.append(float(line.split(' ')[-1]))

npu_total.sort()
print("npu_total time 50: ", npu_total[int(len(npu_total) / 2)])
print("npu_total time 90: ", npu_total[int(len(npu_total) * 0.9)])
print("npu_total time mean: ", np.mean(npu_total))
print("npu_total time max: ", max(npu_total))
print("npu_total time min: ", min(npu_total))
