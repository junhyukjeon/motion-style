import bpy
import math
from mathutils import Vector

# === Utility: Point camera at target ===
def look_at(obj_camera, target_point):
    direction = target_point - obj_camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()

# === Reset Scene ===
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# === Use GPU ===
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
scene.cycles.device = "GPU"
prefs.get_devices()
for d in prefs.devices:
    d.use = True
scene.cycles.samples = 16  # fast test

# === Add Cube ===
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 1))
cube = bpy.context.object
cube.keyframe_insert(data_path="location", frame=1)
cube.location.z = 2
cube.keyframe_insert(data_path="location", frame=2)

# === Add Camera ===
bpy.ops.object.camera_add(location=(0, -6, 2))
camera = bpy.context.object
look_at(camera, Vector((0, 0, 1)))
scene.camera = camera
camera.data.lens = 35  # Wider field of view (35mm)

# === Add Light ===
bpy.ops.object.light_add(type='SUN', location=(5, -5, 5))
light = bpy.context.object
light.data.energy = 1000

# === Add World Brightness ===
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[1].default_value = 1.0

# === Render Settings ===
scene.frame_start = 1
scene.frame_end = 2
scene.render.fps = 2
scene.render.resolution_x = 320
scene.render.resolution_y = 240
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
scene.render.ffmpeg.ffmpeg_preset = 'REALTIME'
scene.render.filepath = "/source/junhyuk/motion-style/final_output.mp4"

# Render!
bpy.ops.render.render(animation=True)
