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
QWen-2.5-VL-7B + 2633 data sft (with-crop-tools) + 2400(steps) * 3(gpus) * 12(generations) * 1(accumulation) [2 iterations]: 10.2%
QWen-2.5-VL-7B + 2633 data sft (with-crop-tools) + 4400(steps) * 3(gpus) * 12(generations) * 1(accumulation) [2 iterations]: 


QWen-2.5-VL-7B + 5650 data sft (with-crop-tools): 12.59% (Warmup)

QWen-2.5-VL-7B + 5650 data sft (with-crop-tools) + rl 2400 steps, 0.01beta :28.72% (Old method)

QWen-2.5-VL-7B + 5650 data sft (with-crop-tools) + rl 3600 steps 0.01beta :21.51% (Old method)

QWen-2.5-VL-7B + 5650 data sft (with-crop-tools) + rl 1200 steps 0.1beta :34.16% (New method)

QWen-2.5-VL-7B + 5650 data sft (with-crop-tools) + rl 2400 steps 0.1beta :31.25% (New method)

QWen-2.5-VL-7B + 5650 data sft (with-crop-tools) + rl 3600 steps 0.1beta :29.48% (New method)

QWen-2.5-VL-7B + 5650 data sft (with-crop-tools) + rl 4800 steps 0.1beta :33.27% (New method)

QWen-2.5-VL-7B + 5650 data sft (with-crop-tools) + rl 6000 steps 0.1beta :31.12% (New method)


UI-TARS-7B + 5650 data sft (with-crop-tools) 3 epoch: 16.82%

UI-TARS-7B + 5650 data sft (with-crop-tools) 6 epoch: 10.50%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 400 steps + 0.02beta + ep0.2-0.28 + ref_sft + grad_1 + new_dataset: No results, 350步左右直接炸

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 2000 steps + 0.02beta + ep0.2-0.28 + ref_sft + grad_1 + new_dataset + sft_loss_0.1_alp: 
稳定不炸, 32.89%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 400 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_4 + new_dataset + sft_loss_0.1_alp (+format selection): 
稳定不炸, 40.04%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 200 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_16 + new_dataset + sft_loss_0.1_alp (+format selection): 
稳定不炸, 45.98%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 400 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_16 + new_dataset + sft_loss_0.1_alp (+format selection): 
稳定不炸, 44.72%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 600 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_16 + new_dataset + sft_loss_0.1_alp (+format selection): 
稳定不炸, 37.07%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 400 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_16 + new_dataset + sft_loss_0.01_alp (+format selection): 
稳定不炸, 38.39%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 200 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_16 + new_dataset + sft_loss_0.01_alp (+format selection): 
稳定不炸, 37.95%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 800 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_16 + new_dataset + sft_loss_0.01_alp (+format selection): 
近900步时炸掉, 37.07%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 200 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_16 + new_dataset + sft_loss_0.8_alp (+format selection): 
稳定不炸, 43.64%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 500 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_16 + new_dataset + sft_loss_0.8_alp (+format selection): 
稳定不炸, 39.47%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 200 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_16 + new_dataset + sft_loss_1_alp (+format selection): 
稳定不炸, 47.88%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 200 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_16 + new_dataset + sft_loss_2_alp (+format selection): 
稳定不炸, 43.01%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 100 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_32 + new_dataset + sft_loss_0.1_alp (+format selection): 
稳定不炸, 36.50%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 100 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_32 + new_dataset + sft_loss_0.1_alp (+format selection): 
稳定不炸, 39.09%

UI-TARS-7B + 2319 data sft (with-crop-tools) 1 epoch + rl 100 steps + 0.00beta + ep0.2-0.28 + ref_sft + grad_4 + new_dataset + sft_loss_0.1_alp (+format selection) + cluster 12 nodes: 
稳定不炸, 37.25%

UI-TARS-7B + 5650 data sft (with-crop-tools) 3 epoch + rl 400 steps +  ???beta + ep??-?? + ref 20 + + grad_1 + old_dataset: 36.31%

UI-TARS-7B + 8612 data sft (with-crop-tools) 3 epoch + rl 1200 steps + 0.02beta + ep0.1-0.2 + ref_sft + grad_1 + old_dataset: 25.62%

UI-TARS-7B + 8612 data sft (with-crop-tools) 3 epoch + rl 400 steps + 0.02beta + ep0.2-0.28 + ref_sft + grad_1 + old_dataset: 23.85%

UI-TARS-7B + 8612 data sft (with-crop-tools) 3 epoch + rl 400 steps + 0.02beta + ep0.2-0.28 + ref_sft + grad_4 + old_dataset:

UI-TARS-7B + 8612 data sft (with-crop-tools) 3 epoch + rl 400 steps + 0.02beta + ep0.2-0.28 + ref_sft + grad_16 + old_dataset: 29.73%

UI-TARS-7B + 12984 data sft (with-crop-tools) 3 epoch + rl 400 steps + 0.02beta + ep0.2-0.28 + ref_sft + grad_16 + old_dataset: