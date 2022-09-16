import os
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt

# Load mesh

def generate_rgb_images(path, dir_name):

    mesh_size = os.path.getsize(path)

    if mesh_size >= 70000000:
        return
    
    mesh = pyrender.Mesh.from_trimesh(trimesh.load(path), smooth=False)
    r = pyrender.OffscreenRenderer(224, 224)
    file_dir,_ = os.path.splitext(path) 

    if not os.path.exists(file_dir):
        os.makedirs(file_dir) 
    #Fancy
    for i in range(25):
        scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0]) # we can also make bg black, looks better
        camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0, aspectRatio=1.0)
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)
        scene.add(mesh, pose=  np.eye(4))

        s2 = 1.0 / np.sqrt(2.0)
        cp = np.eye(4)
        cp[:3,:3] = np.array([
            [0.0, -s2, s2],
            [1.0, 0.0, 0.0],
            [0.0, s2, s2]
        ])
        hfov = np.pi / 6.0
        dist = scene.scale / (2.0 * np.tan(hfov))
        if i == 24:
            cp[:3,3] = dist * np.array([0.7, 0.0, 0.7]) + scene.centroid
            angle = (np.pi / 12.0)
        else:
            cp[:3,3] = dist * np.array([1.0, 0.0, 1.0]) + scene.centroid
            angle= (np.pi / 12.0)*i
        axis = [0,0,1]
        x_rot_mat = trimesh.transformations.rotation_matrix(angle, axis, scene.centroid)
        cp = x_rot_mat.dot(cp)
        scene.add(camera,pose = cp,name="camera")
        scene.add(light, pose=  cp)
        color, _ = r.render(scene,flags=1024)
        filename = f"{file_dir}\\{dir_name}_fancy_{i}.png"
        plt.imsave(filename,color)

    # #Top
    # scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0]) # we can also make bg black, looks better
    # camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0, aspectRatio=1.0)
    # light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)
    # scene.add(mesh, pose=  np.eye(4))
    # scene.add(light, pose=  np.eye(4))
    # cp = np.eye(4)
    # hfov = np.pi / 6.0
    # dist = scene.scale / (2.0 * np.tan(hfov))
    # cp[:3,3] = dist * np.array([0.0, 0.0, 1.0]) + scene.centroid
    # scene.add(camera,pose = cp)
    # color, _ = r.render(scene,flags=1024)
    # filename = f"{file_dir}\\{dir_name}_top.png"
    # plt.imsave(filename,color)

    # #Bottom
    # scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0]) # we can also make bg black, looks better
    # camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0, aspectRatio=1.0)
    # light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)
    # scene.add(mesh, pose=  np.eye(4))
    # scene.add(light, pose=  np.eye(4))
    # cp = np.eye(4)
    # cp[:3,:3] = np.array([
    #     [-1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.0, -1.0]
    # ])
    # hfov = np.pi / 6.0
    # dist = scene.scale / (2.0 * np.tan(hfov))
    # cp[:3,3] = dist * np.array([0.0, 0.0, -1.0]) + scene.centroid
    # scene.add(camera,pose = cp)
    # color, _ = r.render(scene,flags=1024)
    # filename = f"{file_dir}\\{dir_name}_bottom.png"
    # plt.imsave(filename,color)

    # #Back
    # scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0]) # we can also make bg black, looks better
    # camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0, aspectRatio=1.0)
    # light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)
    # scene.add(mesh, pose=  np.eye(4))
    # scene.add(light, pose=  np.eye(4))
    # cp = np.eye(4)
    # cp[:3,:3] = np.array([
    #     [1, 0.0, 0.0],
    #     [0.0, 0.0, -1.0],
    #     [0.0, 1.0, 0.0]
    # ])
    # hfov = np.pi / 6.0
    # dist = scene.scale / (2.0 * np.tan(hfov))
    # cp[:3,3] = dist * np.array([0.0, -1.0, 0.0]) + scene.centroid
    # scene.add(camera,pose = cp)
    # color, _ = r.render(scene,flags=1024)
    # filename = f"{file_dir}\\{dir_name}_back.png"
    # plt.imsave(filename,color)

    # #Front
    # scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0]) # we can also make bg black, looks better
    # camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0, aspectRatio=1.0)
    # light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)
    # scene.add(mesh, pose=  np.eye(4))
    # scene.add(light, pose=  np.eye(4))
    # cp = np.eye(4)
    # cp[:3,:3] = np.array([
    #     [1, 0.0, 0.0],
    #     [0.0, 0.0, 1.0],
    #     [0.0, -1.0, 0.0]
    # ])

    # hfov = np.pi / 6.0
    # dist = scene.scale / (2.0 * np.tan(hfov))
    # #cp[:3,:3] = cp[:3,:3] * rotate_z
    # cp[:3,3] = dist * np.array([0.0, 1.0, 0.0]) + scene.centroid
    # scene.add(camera,pose = cp)
    # color, _ = r.render(scene,flags=1024)
    # filename = f"{file_dir}\\{dir_name}_front.png"
    # color = ndimage.rotate(color, 180)
    # plt.imsave(filename,color)

    # #Side 1
    # scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0]) # we can also make bg black, looks better
    # camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0, aspectRatio=1.0)
    # light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)
    # scene.add(mesh, pose=  np.eye(4))
    # scene.add(light, pose=  np.eye(4))
    # cp = np.eye(4)
    # cp[:3,:3] = np.array([
    #     [0.0000000,  0.0000000,  1.0000000],
    #     [0.0000000,  1.0000000,  0.0000000],
    #     [-1.0000000,  0.0000000,  0.0000000]
    # ])

    # hfov = np.pi / 6.0
    # dist = scene.scale / (2.0 * np.tan(hfov))
    # cp[:3,3] = dist * np.array([1.0, 0.0, 0.0]) + scene.centroid
    # scene.add(camera,pose = cp)
    # color, _ = r.render(scene,flags=1024)
    # filename = f"{file_dir}\\{dir_name}_side1.png"
    # color = ndimage.rotate(color, 270)
    # plt.imsave(filename,color)

    # #Side 2
    # scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0]) # we can also make bg black, looks better
    # camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0, aspectRatio=1.0)
    # light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)
    # scene.add(mesh, pose=  np.eye(4))
    # scene.add(light, pose=  np.eye(4))
    # cp = np.eye(4)
    # cp[:3,:3] = np.array([
    #     [0.0000000,  0.0000000,  -1.0000000],
    #     [0.0000000,  1.0000000,  0.0000000],
    #     [1.0000000,  0.0000000,  0.0000000]
    # ])

    # hfov = np.pi / 6.0
    # dist = scene.scale / (2.0 * np.tan(hfov))
    # cp[:3,3] = dist * np.array([-1.0, 0.0, 0.0]) + scene.centroid
    # scene.add(camera,pose = cp)
    # color, _ = r.render(scene,flags=1024)
    # filename = f"{file_dir}\\{dir_name}_side2.png"

    # color = ndimage.rotate(color, 90)

    # plt.imsave(filename,color)
    r.delete()

dataset_path = '../data/ModelNet40'

for root, dirs, files in os.walk(dataset_path):
        for dir in dirs:
            print(dir)
        for name in files:
            filename = os.path.join(root, name)
            file_dir,filetype = os.path.splitext(filename)
            if filetype == '.off':
                split = file_dir.split('\\')
                N = len(split)
                generate_rgb_images(filename, split[N - 1])
generate_rgb_images('../data/ModelNet40/xbox/test/xbox_0104.off', 'xbox_0104')