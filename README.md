# Efficient Multi-agent Communication via Shapley Message Value

This is the official implementation for our paper [Efficient Multi-agent Communication via Shapley Message Value](https://www.ijcai.org/proceedings/2022/82), based on the open-sourced codebases [PyMARL2](https://github.com/hijkzzz/pymarl2) which is built on the codebase [PyMARL](https://github.com/oxwhirl/pymarl).

## Installation instructions

```
conda create -n sms python==3.7
conda activate sms
pip install -r requirements.txt
```

Set up listener-speaker environment, which is developed based on the open-sourced repo [lb-foraging](https://github.com/semitable/lb-foraging):

```shell
cd listener_speaker
pip install -e .
```

## Run an experiment 

```shell
python3 src/main.py --config=[Algorithm name] --env-config=[Env name] with [key1=value1] [key2=value2] ...
```

To run an exiperiment, you need to provide the config files for both the algorithm and environment. These files act as default configurations for them and are located in `src/config`.
`--config` refers to the config files in `src/config/algs`, which includes SMS and some other baselines. `--env-config` refers to the config files in `src/config/envs`.
You can mofidy the hyper-parameters for an experiment by specifying them after the `with` keyword.
Once an experiment is done, the results will be stored in the `results` folder.

For example, to run SMS on the listener-speaker environment with $4$ listener-speaker pairs on a $15\times 15$ grid world:

```
python3 src/main.py \
    --config=sms \
    --env-config=listener_speaker \
    with \
    env_args.key="listener_speaker:ListenerSpeaker-15x15-4-v1" \
    msg_dim=2 \
    entropy_coef=0.0 \
    start_msg_select_timestep=200000 \
```

Run SMS on the task hallway:

```
python3 src/main.py \
    --config=sms \
    --env-config=hallway \
    with \
    start_msg_select_timestep=300000 \
```

We also provide corresponding scripts to run the experiments:
```
sh scripts/run_ls_sms.sh
sh scripts/run_hw_sms.sh
```