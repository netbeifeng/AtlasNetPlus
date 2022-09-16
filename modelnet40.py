
import os
import open3d
import numpy as np
import random
import torch
import normalization as PointCloudPreProcess
from PIL import Image
from torchvision import transforms

# Dataloader for ModelNet40
class ModelNet40AtlasNet(torch.utils.data.Dataset):
    def __init__(self, split = 'train', class_choice = None, 
                    shuffle = False, pcd_normalization = None, 
                    img_normalization = False, num_point = 2048, inference = False, mvcnn = False):
        self.inference = inference
        self.mvcnn = mvcnn
        self.num_point = num_point
        self.class_choice = class_choice
        self.split = split
        self.shuffle = shuffle
        self.pcd_normalization = pcd_normalization
        self.img_normalization = img_normalization

        self.place_holder_path = "data/ModelNet40_placeholder"
        self.pcd_path = "data/ModelNet40_pointclouds"
        self.rendering_path = "data/ModelNet40_renderings"

        self.classes = sorted(os.listdir(self.pcd_path))
        self.all_list = self.get_list_all()

        if self.split == "test" and self.img_normalization:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        elif self.split == "train" and self.img_normalization:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else: 
            self.transform = transforms.ToTensor()


    def __getitem__(self, index):
        path = self.all_list[index]
        pcd_file_path = path.replace(self.place_holder_path, self.pcd_path) + ".pcd"
        img_folder_path = path.replace(self.place_holder_path, self.rendering_path)
        pcd = self.read_pcd(pcd_file_path)
        img, choosen_filename, all_imgs = self.read_sample_rendering(img_folder_path)

        splited = pcd_file_path.split('/')

        operation = PointCloudPreProcess.Normalization(pcd, keep_track=True)

        if self.pcd_normalization is not None:
            if self.pcd_normalization == "UnitBall":
                operation.normalize_unitL2ball()
            elif self.pcd_normalization == "BoundingBox":
                operation.normalize_bounding_box()
            else:
                pass
        
        if self.inference:
            dict = {
                'class': splited[2],
                'class_id': self.classes.index(splited[2]),
                'pcd_filename': splited[4],
                'img_filename': choosen_filename,
                'operation': operation,
                'all_renderings': all_imgs,
                'rendering': img,
                'pointcloud': pcd
            }
        else:
            if self.mvcnn:
                dict = {
                    'class': splited[2],
                    'all_renderings': all_imgs,
                    'class_id': self.classes.index(splited[2]),
                    'pointcloud': pcd
                }
            else: 
                dict = {
                    'class': splited[2],
                    'class_id': self.classes.index(splited[2]),
                    'rendering': img,
                    'pointcloud': pcd
                }

        return dict

    def __len__(self):
        return len(self.all_list)

    def get_list_all(self):
        ret_list = []
        if self.class_choice is None:
            for a_class in self.classes:
                ret_list.append(self.get_list_of_class(a_class))
        else:
            for a_class in self.class_choice:
                ret_list.append(self.get_list_of_class(a_class))
                

        if self.shuffle:
            flat_list = list(np.concatenate(ret_list).flat)
            return random.sample(flat_list, len(flat_list))
        else: 
            return list(np.concatenate(ret_list).flat)

    def get_list_of_class(self, class_name):
        file_list = sorted(os.listdir(self.rendering_path + "/" + class_name + "/" + self.split))
        new_file_list = [f"{self.place_holder_path}/{class_name}/{self.split}/{file_name}" for file_name in file_list]

        if self.shuffle:
            return random.sample(new_file_list, len(new_file_list))
        else:   
            return new_file_list

    def read_pcd(self, pcd_path):
        pcd = open3d.io.read_point_cloud(pcd_path)
        np_pcd = np.asarray(pcd.points)
        size = np_pcd.shape[0]
        if self.num_point <= np_pcd.shape[0]:
            sample_list = np.random.choice(size, self.num_point, replace=False)
            np_pcd = np_pcd[sample_list, :]
            # Not that good, just discard the rest
            # np_pcd = np_pcd[:self.num_point, :]
        
        torch_pcd = torch.tensor(np_pcd)
        return torch_pcd

    def read_sample_rendering(self, foldername):
        rendering_list = sorted(os.listdir(foldername))


        if self.inference or self.mvcnn:
            all_images = []
            for i in rendering_list:
                image = Image.open(foldername+ "/" + i)
                image = image.convert('RGB')
                image = self.transform(image)
                all_images.append(image)
                
            if self.split == "test":
                choosen_one = rendering_list[0]
                image = all_images[0]
            else:
                choosen_idx = random.randint(0, len(rendering_list) - 1)
                choosen_one = rendering_list[choosen_idx]
                image = all_images[choosen_idx]

            all_images = torch.stack(all_images, 0)
            return image, choosen_one, all_images
        else:
            if self.split == "test":
                choosen_one = rendering_list[0]
            else:
                choosen_one = np.random.choice(rendering_list)
            fullpath = foldername + "/" + choosen_one
            image = Image.open(fullpath)
            image = image.convert('RGB')
            image = self.transform(image)
            return image, choosen_one, []
            
    
class ModelNet40MultiView(torch.utils.data.Dataset):
    def __init__(self, split="train", scale_aug=False, rot_aug=False, \
                 num_views=12, shuffle=True):
        self.rendering_path = "data/ModelNet40_renderings"
        self.split = split
        self.shuffle = shuffle
        self.classes = sorted(os.listdir(self.rendering_path))
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.num_views = num_views
        self.all_list = sorted(self.get_list_all())
        
        if self.shuffle is True:
            self.all_list = random.sample(self.all_list, len(self.all_list))
        if self.split == "test":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        
    def __len__(self):
        return len(self.all_list)
    
    def __getitem__(self, idx):
        sample = self.all_list[idx]
        class_name = sample.split("/")[2]
        class_id = self.classes.index(class_name)
        
        imgs = []
        imgs_path = []
        for view in sorted(os.listdir(sample)):
            view = view.decode('UTF-8')
            im = Image.open(sample + "/" + view).convert('RGB')
            imgs_path.append(sample + "/" + view)
            if self.transform:
                im = self.transform(im)
            imgs.append(im)
        
        return (class_id, torch.stack(imgs), imgs_path)
            
    def get_list_all(self):
        ret_list = []
        for a_class in self.classes:
            ret_list.append(self.get_list_of_class(a_class))
                
        if self.shuffle:
            flat_list = list(np.concatenate(ret_list).flat)
            return random.sample(flat_list, len(flat_list))
        else: 
            return list(np.concatenate(ret_list).flat)
        
    def get_list_of_class(self, class_name):
        file_list = sorted(os.listdir(self.rendering_path + "/" + class_name + "/" + self.split))
        new_file_list = [f"{self.rendering_path}/{class_name}/{self.split}/{file_name}" for file_name in file_list]

        if self.shuffle:
            return random.sample(new_file_list, len(new_file_list))
        else:   
            return new_file_list
        
class ModelNet40SingleView(torch.utils.data.Dataset):
    def __init__(self, split="train", scale_aug=False, rot_aug=False, num_views=12, shuffle=False):
        self.rendering_path = "data/ModelNet40_renderings"
        self.split = split
        self.shuffle = shuffle
        self.classes = sorted(os.listdir(self.rendering_path))
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.num_views = num_views
        self.all_list = sorted(self.get_list_all())
        
        self.all_img_list = []
        for img_folder in self.all_list:
            imgs_in_folder = sorted(os.listdir(img_folder))
            full_paths = [f'{img_folder}/{str(name, "utf-8")}' for name in imgs_in_folder]
            self.all_img_list.append(full_paths) 
            
        def flatten(l):
            return [item for sublist in l for item in sublist]
        
        self.all_img_list = flatten(self.all_img_list)
        if self.shuffle is True:
            self.all_img_list = random.sample(self.all_img_list, len(self.all_img_list))   
            
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])    
        
    def get_list_all(self):
        ret_list = []
        for a_class in self.classes:
            ret_list.append(self.get_list_of_class(a_class))
                
        if self.shuffle:
            flat_list = list(np.concatenate(ret_list).flat)
            return random.sample(flat_list, len(flat_list))
        else: 
            return list(np.concatenate(ret_list).flat)
        
    def get_list_of_class(self, class_name):
        file_list = sorted(os.listdir(self.rendering_path + "/" + class_name + "/" + self.split))
        new_file_list = [f"{self.rendering_path}/{class_name}/{self.split}/{file_name}" for file_name in file_list]

        if self.shuffle:
            return random.sample(new_file_list, len(new_file_list))
        else:   
            return new_file_list
        
    def __len__(self):
        return len(self.all_img_list)
    
    def __getitem__(self, idx):
        path = self.all_img_list[idx]
        class_name = path.split('/')[2]
        class_id = self.classes.index(class_name)
        im = Image.open(path).convert('RGB')
        if self.transform:
            im = self.transform(im)
            
        return (class_id, im, path)

        
if __name__ == '__main__':
    # load airplane class only
    dataset = ModelNet40AtlasNet(shuffle=False, pcd_normalization="BoundingBox", img_normalization=False)
    print(dataset.__len__())
    print(dataset[0])
    
    dataset = ModelNet40MultiView()
    print(dataset[0])
    
    dataset = ModelNet40SingleView()
    print(dataset[0])
    print(dataset.__len__()/12)