import os

dataset_path = 'ModelNet40'

for root, dirs, files in os.walk(dataset_path):
        for name in files:
            filename = os.path.join(root, name)
            file_dir,filetype = os.path.splitext(filename)
            #print(filename)
            if filetype == '.off':
                lines = []
                with open(filename) as f:
                    lines = f.readlines()
                lines[0] = lines[0].replace("OFF","OFF\n/newline")
                off, rest = lines[0].split("/newline")
                lines[0] = off
                lines.insert(1, rest)
                with open(filename, 'w') as f:
                    lines = f.writelines(lines)
