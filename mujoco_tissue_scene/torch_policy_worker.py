"""Pi0.5 inference worker in its original training environment; JSON-line IPC."""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors


def main(checkpoint):
    path=Path(checkpoint)
    if not (path/"model.safetensors").is_file():
        raise FileNotFoundError(path/"model.safetensors")
    torch.manual_seed(1000)
    torch.set_num_threads(4)
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured):
        policy=PI05Policy.from_pretrained(str(path),local_files_only=True,strict=True)
    log=captured.getvalue()
    print(log,file=sys.stderr,flush=True)
    # The vendor loader catches exceptions and can return uninitialized weights.
    # Never accept that fallback as a trained model.
    if "All keys loaded successfully!" not in log:
        raise RuntimeError("Checkpoint strict load not confirmed; refusing uninitialized policy")
    policy.config.device="cuda"
    policy.config.gradient_checkpointing=False
    policy.to("cuda").eval()
    preprocessor,postprocessor=make_pre_post_processors(
        policy_cfg=policy.config,pretrained_path=str(path),
        preprocessor_overrides={"device_processor":{"device":"cuda"}})
    policy.reset()
    print("READY "+json.dumps({"type":policy.config.type,"checkpoint":str(path),
          "device":"cuda","strict_load":True,"chunk_size":policy.config.chunk_size}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get("stop"):
            break
        with np.load(request["observation_path"],allow_pickle=False) as data:
            state=torch.from_numpy(data["state"].copy())
            batch={"observation.state":state,"task":request["task"]}
            for source,key in [("central","observation.images.base_0_rgb"),
                               ("left_wrist","observation.images.left_wrist_0_rgb")]:
                batch[key]=torch.from_numpy(data[source].copy()).permute(2,0,1).float()/255.
        started=time.monotonic()
        with torch.inference_mode():
            processed=preprocessor(batch)
            _,masks=policy._preprocess_images(processed)
            mask_values=[mask.cpu().tolist() for mask in masks]
            if mask_values != [[True],[True],[False]]:
                raise RuntimeError(f"Unexpected camera masks: {mask_values}")
            actions=policy.predict_action_chunk(processed)
            actions=postprocessor(actions).detach().float().cpu().numpy()
        if actions.shape != (1,50,32) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid actions: {actions.shape}")
        result={"id":request["id"],"actions":actions[0].tolist(),
                "inference_seconds":time.monotonic()-started,"camera_masks":mask_values,
                "cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
        print("RESULT "+json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    main(parser.parse_args().checkpoint)
