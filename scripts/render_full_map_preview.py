import mujoco
from PIL import Image
from robocon_2027_full_map import build_full_map
xml,_=build_full_map()
model=mujoco.MjModel.from_xml_path(str(xml))
model.vis.global_.offwidth=1600
model.vis.global_.offheight=1000
data=mujoco.MjData(model)
mujoco.mj_forward(model,data)
renderer=mujoco.Renderer(model,height=900,width=1500)
cam=mujoco.MjvCamera()
cam.type=mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:]=(17.2,0,0)
cam.distance=48
cam.azimuth=90
cam.elevation=-76
renderer.update_scene(data,camera=cam)
output=xml.parent/'full_map_overview.png'
Image.fromarray(renderer.render()).save(str(output))
renderer.close()
print(output)
