"""A trainer checkpoint (.ckpt) as the adapter file the rooms load: the LoRA tensors at the precision they trained
at, with their lora_config embedded as stable-audio-tools reads it. Run in the engine's environment."""

import json
import sys


def main(src: str, dst: str) -> int:
    from safetensors.torch import save_file
    from stable_audio_tools.models.lora.utils import load_lora_checkpoint

    state, config = load_lora_checkpoint(src)
    tensors = {k: v.detach().contiguous().float() for k, v in state.items()}
    save_file(tensors, dst, metadata={"lora_config": json.dumps(config)})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
