#!/bin/bash

#挑选随机结构，生成对应的POSCAR供VASP计算


cat << EOF > temp.py

import dpdata
import glob
import os
import numpy as np
import shutil

#使用dpdata处理数据，将lammps轨迹输入到dpdata中
data = dpdata.System('010.dump',fmt='lammps/dump')
print (data)

#确定要验证的结构数目，例：这里总步数10000steps，随机抽取两帧
frame_idx = np.random.choice(10000,size=2,replace=False)

#生成最初版本的POSCAR，此时的POSCAR元素种类显示错误
for idx in frame_idx:
    data.to("vasp/POSCAR", "POSCAR_{}".format(idx), frame_idx=idx)


#更改元素种类
# Find all files that match the pattern 'POSCAR_*'
for filename in glob.glob('POSCAR_*'):
    # Open the original file in read mode and a new file in write mode
    with open(filename, 'r') as infile, open(filename + '_new', 'w') as outfile:
        # Loop over each line in the file
        for line in infile:
            # Replace 'TYPE_0' with 'O' and 'TYPE_1' with 'I' in the line
            new_line = line.replace('TYPE_0', 'O').replace('TYPE_1', 'H').replace('TYPE_2', 'C').replace('TYPE_3', 'N').replace('TYPE_4', 'Pb').replace('TYPE_5', 'I')
            # Write the new line to the new file
            outfile.write(new_line)

if not os.path.exists('POSCAR-VASP'):
    os.makedirs('POSCAR-VASP')

for idx in frame_idx:
    subdir = 'POSCAR-VASP/{}'.format(idx)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    shutil.move("POSCAR_{}_new".format(idx), subdir)


EOF

python temp.py
rm temp.py

rm POSCAR_*
for f in POSCAR-VASP/*/POSCAR_*; do mv -- "$f" "${f%/*}/POSCAR"; done
for d in POSCAR-VASP/*/; do cp INCAR KPOINTS "$d"; done
