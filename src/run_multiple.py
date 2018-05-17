

from subprocess import Popen
from itertools import product
import os

env_config = "multi_obj"
env_ext = "10obj_every2"

model_to_test = ['resnet_dqn', 'dqn_filmed']
#extension_to_test = ['soft_update0_1', 'soft_update0_01']
extension_to_test = ['soft_update0_1', 'soft_update0_01', 'soft_update0_001', 'soft_update0_0001',
                      'hard_update0_1', 'hard_update0_01', 'hard_update0_001']

processes = []

for test_num, (model, model_ext) in enumerate(product(model_to_test,extension_to_test)):

    if test_num > 15:
        "Can't run more than 16 expes in parallel"
        break

    device_to_use = test_num%4

    config_location = 'config/{}/{}'
    env_config_path = config_location.format("env_base", env_config)
    env_ext_path = config_location.format("env_ext", env_ext)

    model_config_path = config_location.format("model", model)
    model_ext_path = config_location.format("model_ext", model_ext)

    command = "python3 src/main.py -env_config {}.json -env_extension {}.json -model_config {}.json -model_extension {}.json -device {}".format(
                env_config_path, env_ext_path, model_config_path, model_ext_path, device_to_use)
    print(command)
    processes.append(Popen(command, shell=True, env=os.environ.copy()))

for p in processes:
    p.wait()
print('Done running experiments')

out_dir = "out/"+env_config+'_'+env_ext
results = []

# config_files.txt  config.json  eval_curve.png  last_10_std_length  last_10_std_reward  last_5_length.npy  last_5_reward.npy
# length.npy  mean_length  mean_reward  model_name  reward.npy  train_lengths  train.log  train_rewards

for subfolder in os.listdir(out_dir):
    result_path = out_dir+'/'+subfolder+'/'
    print(result_path)
    name = open(result_path+"model_name", 'r').read()
    score = float(open(result_path+"mean_length", 'r').read())

    results.append((name, subfolder, score))

results.sort(key=lambda x:x[2])
print(results)

summary_str = ''
for name, subfolder, length in results:
    summary_str += "{} {} {}\n".format(name, subfolder, length)

open(out_dir+"/summary", 'w').write(summary_str)