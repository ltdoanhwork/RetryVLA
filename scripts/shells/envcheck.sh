cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA && source env/activate.sh
python -c "
import cv2, torch, mujoco; print('cv2', cv2.__version__, 'cuda', torch.cuda.is_available())
from retryvla.envs.libero_env import LiberoSim, discover_scenes
import time; s=discover_scenes('data/assets/scenes/scenes/selected_bddl_scenes_spatial')[0]; sim=LiberoSim(); r=sim.load_scene(s.bddl,s.npy,True)
t=time.time(); [sim.step([0]*7) for _ in range(50)]; print('50 steps', round(time.time()-t,2))
from OpenGL import GL; print('GL', GL.glGetString(GL.GL_RENDERER))
" 2>&1 | grep -E "^cv2|steps|^GL|Error" | grep -v EGLError
