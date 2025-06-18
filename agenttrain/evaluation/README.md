# Experiment Results

UI-TARS-1.5-7B (zero-shot): 40.42%
UI-TARS-1.5-7B (with-crop-tools): 17.08%

QWen-2.5-VL-7B (zero-shot): 5.58%
QWen-2.5-VL-7B (with-crop-tools): 0.57%

sft Trained Models
QWen-2.5-VL-7B + 2633 data sft (with-crop-tools): 3.61%
UI-TARS-1.5-7B + 2633 data sft (with-crop-tools): 5.31%

QWen-2.5-VL-7B + 75456 data sft (with-crop-tools): 2.02%

RL Trained Models:
QWen-2.5-VL-7B + 75456 data sft (with-crop-tools) + 800(steps) * 3(gpus) * 12(generations) * 1(accumulation) [2 iterations]: 2.59%
QWen-2.5-VL-7B + 75456 data sft (with-crop-tools) + (800+1000)(steps, possible data overlap) * 3(gpus) * 12(generations) * 1(accumulation) [2 iterations]: 1.90%
QWen-2.5-VL-7B + 2633 data sft (with-crop-tools) + 4000(steps) * 6(gpus) * 4(generations) * 1(accumulation) [2 iterations]: 2.02%
QWen-2.5-VL-7B + 2633 data sft (with-crop-tools) + 4200(steps) * 6(gpus) * 4(generations) * 1(accumulation) [2 iterations]: 2.78%
QWen-2.5-VL-7B + 2633 data sft (with-crop-tools) + 4400(steps) * 6(gpus) * 4(generations) * 1(accumulation) [2 iterations]: 3.92%
QWen-2.5-VL-7B + 2633 data sft (with-crop-tools) + 4600(steps) * 6(gpus) * 4(generations) * 1(accumulation) [2 iterations]: 2.85%