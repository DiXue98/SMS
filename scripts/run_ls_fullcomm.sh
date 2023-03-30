python3 src/main.py \
    --config=fullcomm \
    --env-config=listener_speaker \
    with \
    env_args.key="listener_speaker:ListenerSpeaker-15x15-4-v1" \
    msg_dim=2 \
    use_tensorboard=True \
    entropy_coef=0.0 \
    name=fullcomm_ls