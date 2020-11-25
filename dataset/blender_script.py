import argparse, sys, os
import mathutils
import numpy as np
import bpy
from math import radians
from bpy_extras.object_utils import world_to_camera_view

parser = argparse.ArgumentParser(description='Renders given obj file by rotating a camera around it.')
parser.add_argument('--obj', type=str,help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='/tmp',help='The path the output will be dumped to.')
parser.add_argument('--res', type=int, default='64', help='Resolution of the output images')
parser.add_argument('--camera_path', type=str, default='./camPosListDodecAzEl.txt', help="path to az and el angles of the camera positions")
parser.add_argument('--clip_start', type=float, default=1.5)
parser.add_argument('--clip_end', type=float, default=3.5)
parser.add_argument('--cam_type', type=str, default='PERSP', help="PERSP or ORTHO")

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


def read_camera_positions(path):
	camera_azel = []
	with open(path) as f:
		for vp, line in enumerate(f.readlines()):
			line = line.strip().split()
			az, el = line
			camera_azel.append([float(az), float(el)])
	return camera_azel


# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_depth = '8'

# Clear default nodes
for n in tree.nodes:
	tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')
depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'

# Map the depth values to range [0,1]
map = tree.nodes.new(type="CompositorNodeMapRange")
map.inputs[1].default_value = args.clip_start
map.inputs[2].default_value = args.clip_end
map.inputs[3].default_value = 0
map.inputs[4].default_value = 1
links.new(render_layers.outputs['Depth'], map.inputs[0])
links.new(map.outputs[0], depth_file_output.inputs[0])


# Delete default cube and import the model
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()
bpy.ops.import_scene.obj(filepath=args.obj, split_mode="OFF", axis_forward='-X')

scene = bpy.context.scene
obj = scene.objects['model_normalized']

# Make light just directional, disable shadows.
lamp = bpy.data.lamps['Lamp']
lamp.type = 'SUN'
lamp.shadow_method = 'NOSHADOW'
lamp.use_specular = False  # disable specular shading:

scene = bpy.context.scene
scene.render.resolution_x = args.res
scene.render.resolution_y = args.res
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'
model_identifier = os.path.split(args.obj)[0].split('/')[-2]


camera_azel = read_camera_positions(args.camera_path)
for output_node in [depth_file_output]:
	output_node.base_path = ''

# Set camera values
cam = scene.camera
cam.location = [2.5, 0, 0]
cam.rotation_mode = 'XYZ'
cam.rotation_euler = [0.0, radians(90), 0]
if args.cam_type=='ORTHO':
	cam.data.type='ORTHO'
	cam.data.ortho_scale = 1
cam.data.sensor_width = 30
cam.data.sensor_height = 30
cam.data.angle = 40*np.pi/180
cam.data.clip_start = args.clip_start
cam.data.clip_end = args.clip_end
bpy.context.scene.update()


if not os.path.exists(os.path.join(args.output_folder, model_identifier)):
	os.mkdir(os.path.join(args.output_folder, model_identifier))

# Render depthmaps for all 20 viewpoints
for i, cam_pos in enumerate(camera_azel):
	filename = 'depth_' + str(i).zfill(2)
	depth_file_output.file_slots[0].path = os.path.join(args.output_folder, model_identifier, filename)
	az, el = cam_pos
	obj.rotation_mode = 'YXZ'
	obj.rotation_euler = [0.0, np.pi / 2 - az, -el]
	bpy.context.scene.update()
	bpy.ops.render.render(write_still=False, use_viewport=True)
