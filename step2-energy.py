import re
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt	


#######VASP#######

#匹配VASP OUTCAR中的体系总能量
dft_energy = r'energy  without entropy= (\s+-?\d+\.\d+)  energy\(sigma->0\) = (?P<energy>\s+-?\d+\.\d+)'

# 获取文件夹名作为数字的函数
def get_numeric_folder_name(path):
    folder_name = os.path.basename(os.path.dirname(path))
    return int(folder_name)

def vasp_energies(directory): 
    vasp_energies = {}

    outcar_files = sorted(glob.glob(directory + '/**/OUTCAR', recursive=True), key=get_numeric_folder_name)
    
    for outcar_file in outcar_files:
        with open(outcar_file, 'r') as f:
            for line in f:
                match = re.search(dft_energy, line)
                if match:
                    energy = match['energy']
                    step = get_numeric_folder_name(outcar_file)
                    vasp_energies[step] = float(energy)
    return vasp_energies

#######VASP#######

#######LAMMPS#######

#匹配lammps输出的log.lammps中的能量字段
def lammps_energies(lammpslog_file):
    energy_log = re.compile(
    r"""
        \s*(?P<step>\d+)            # Step
        \s+(?P<pot_e>[-]?\d+\.?\d*) # PotEng
        \s+[-]?\d+\.?\d*            # KinEng
        \s+(?P<tot_e>[-]?\d+\.?\d*) # TotEng
        \s+[-]?\d+\.?\d*            # Temp
        \s+[-]?\d+\.?\d*            # Press
        \s+[-]?\d+\.?\d*            # Volume
        \s*\n                       # Match any trailing whitespace until the newline
    """,
    re.VERBOSE
    )
#储存进一个{}中
    lammps_energies = {}
    with open(lammpslog_file, 'r') as f:
        content = f.read()
        for match in energy_log.finditer(content):
            step = int(match['step'])
            pot_e = float(match['pot_e'])
            lammps_energies[step] = pot_e
    
    return lammps_energies

#######LAMMPS#######

#使用时需修改VASP和LAMMPS输出文件的路径以及原子个数
vasp_out = './VASP-confirm/'
vasp_energies = vasp_energies(vasp_out)
lammps_out = './lammps-thermo1/5ps/log.lammps'
lammps_energies = lammps_energies(lammps_out)
atoms = 222

print("VASP Energies:")
for step, energy in vasp_energies.items():
    print(f"Step: {step}, Energy: {energy}")

# Normalize the energies to per atom
vasp_energies_per_atom = {k: v / atoms for k, v in vasp_energies.items()}
lammps_energies_per_atom = {k: v / atoms for k, v in lammps_energies.items()}

# ...

# 匹配VASP和LAMMPS步骤并提取能量
x = []
y = []
for step, vasp_energy in vasp_energies_per_atom.items():
    lammps_energy = lammps_energies_per_atom.get(step + 1)
    if lammps_energy is not None:
        x.append(lammps_energy)
        y.append(vasp_energy)

# 计算MSE和RMSE
mse = mean_squared_error(y, x)
rmse = sqrt(mse)

print(f"MSE: {mse}, RMSE: {rmse}")

# 生成散点图
plt.rcParams["font.family"] = "Arial"
plt.figure(figsize=(12, 12))
plt.scatter(x, y, color='orange',alpha=0.8)

# Add y=x line
lims = [np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()])]  # max of both axes

plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
plt.xlim(lims)
plt.ylim(lims)

plt.xlabel('DPMD Energy (eV/atom)', fontsize=26)
plt.ylabel('DFT Energy (eV/atom)', fontsize=26)
plt.tick_params(axis='both', labelsize=22)
for axis in ['top','bottom','left','right']:
  plt.gca().spines[axis].set_linewidth(2)

plt.savefig("energy.tiff", dpi=300)
plt.show()
