cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA && source env/activate.sh
python - <<'PY' 2>&1 | grep -E "^OK|Error" | grep -v EGLError
import cv2, torch
from torchcodec.decoders import VideoDecoder
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/libero", episodes=[1261])
x = ds[0]
print("OK video decode", tuple(x["observation.images.image"].shape), "cuda", torch.cuda.is_available())
from retryvla.envs.libero_env import LiberoSim, discover_scenes
import time; s = discover_scenes("data/assets/scenes/scenes/selected_bddl_scenes_spatial")[0]
sim = LiberoSim(); sim.load_scene(s.bddl, s.npy, True); t = time.time(); [sim.step([0]*7) for _ in range(50)]
print("OK sim 50 steps", round(time.time() - t, 2))
PY
