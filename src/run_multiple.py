from subprocess import Popen
from itertools import product
import os

env_config = "multi_obj_test"
env_ext = "10obj_every2"

model_to_test = ['resnet_dqn']
extension_to_test = ['soft_update0_01', 'soft_update0_1']

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
