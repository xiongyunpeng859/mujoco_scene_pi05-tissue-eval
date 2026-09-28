import yaml
c = yaml.safe_load(open('/workspace/shared/mujoco_tissue_scene/configs/scene.yaml'))
print('friction      ', c['boxes'][0]['friction'])
print('arm position  ', c['arm']['position'])
print('arm euler     ', c['arm']['euler'])
print('geom margin   ', c['table'].get('geom_margin_xy'))
print('tray centre   ', c['tray']['center'], 'yaw', c['tray']['yaw'])
