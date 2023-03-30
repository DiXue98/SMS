python3 src/main.py \
    --config=sms \
    --env-config=listener_speaker \
    with \
    env_args.key="listener_speaker:ListenerSpeaker-15x15-4-v1" \
    msg_dim=2 \
    start_msg_select_timestep=200000 \
    entropy_coef=0.0 \
    use_tensorboard=True \
    name=sms_ls