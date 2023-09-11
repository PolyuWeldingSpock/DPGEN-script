import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import pandas as pd

#######VASP###########

def get_numeric_folder_name(path):
    folder_name = os.path.basename(os.path.dirname(path))
    return int(folder_name)

def vasp_forces(directory):
    block_regex = re.compile(
        r"""
        POSITION\s+TOTAL-FORCE\s\(eV\/Angst\)\n
        \s\-+\n
        (?P<block>
            (
                \s+\d+\.\d+
                \s+\d+\.\d+
                \s+\d+\.\d+
                \s+[\s-]\d+\.\d+
                \s+[\s-]\d+\.\d+
                \s+[\s-]\d+\.\d+
            \n)+
        )
        """,
        re.VERBOSE
    )

    line_regex = re.compile(
        r"""
        \s+(?P<x_cor>\d+\.\d+)
        \s+(?P<y_cor>\d+\.\d+)
        \s+(?P<z_cor>\d+\.\d+)
        \s+(?P<x_force>[\s-]\d+\.\d+)
        \s+(?P<y_force>[\s-]\d+\.\d+)
        \s+(?P<z_force>[\s-]\d+\.\d+)
        """,
        re.VERBOSE
    )

    dft_forces = {}
    outcar_files = sorted(glob.glob(directory + '/**/OUTCAR', recursive=True), key=get_numeric_folder_name)
    
    for outcar_file in outcar_files:
        with open(outcar_file, 'r') as f:
            content = f.read()
            for block_match in block_regex.finditer(content):
                step = get_numeric_folder_name(outcar_file)
                forces_for_step = []
                
                for line_match in line_regex.finditer(block_match.group('block')):
                    x = line_match.group('x_cor')
                    y = line_match.group('y_cor')
                    z = line_match.group('z_cor')
                    fx = line_match.group('x_force')
                    fy = line_match.group('y_force')
                    fz = line_match.group('z_force')

                    forces_for_step.append({
                        'position': np.array([float(x), float(y), float(z)]),
                        'force': np.array([float(fx), float(fy), float(fz)])
                    })
                dft_forces[step] = forces_for_step
    num_steps = len(dft_forces)
    print(f"There are {num_steps} steps in the DFT forces data.")
    return dft_forces


#######VASP###########

#######LAMMPS##########

def lammps_forces(directory):
    block_regex = re.compile(
        r"""
        ITEM\:\s+TIMESTEP\n
        (?P<md_step>\d+)\n
        ITEM\:\sNUMBER\sOF\sATOMS\n
        \d+\n
        ITEM\:\sBOX\sBOUNDS\spp\spp\spp\n
        \d+\.\d+e[+-]\d+\s+\d+\.\d+e[+-]\d+\n
        \d+\.\d+e[+-]\d+\s+\d+\.\d+e[+-]\d+\n
        \d+\.\d+e[+-]\d+\s+\d+\.\d+e[+-]\d+\n
        ITEM\:\sATOMS\sid\stype\selement\sx\sy\sz\sfx\sfy\sfz\n
        (?P<block>
            (
                \d+
                \s+\d+
                \s+[A-Z][a-z]?
                \s+-?\d+(\.\d+)?(e-?\d+)?
                \s+-?\d+(\.\d+)?(e-?\d+)?
                \s+-?\d+(\.\d+)?(e-?\d+)?
                \s+-?\d+(\.\d+)?(e-?\d+)?
                \s+-?\d+(\.\d+)?(e-?\d+)?
                \s+-?\d+(\.\d+)?(e-?\d+)?               
                \n
            )+
        )
        """,
        re.VERBOSE
    )

    line_regex = re.compile(
        r"""
        (?P<ATOMS>\d+)
        \s+\d+
        \s+[A-Z][a-z]?
        \s+(?P<x>-?\d+(\.\d+)?(e-?\d+)?)
        \s+(?P<y>-?\d+(\.\d+)?(e-?\d+)?)
        \s+(?P<z>-?\d+(\.\d+)?(e-?\d+)?)
        \s+(?P<fx>-?\d+(\.\d+)?(e-?\d+)?)
        \s+(?P<fy>-?\d+(\.\d+)?(e-?\d+)?)
        \s+(?P<fz>-?\d+(\.\d+)?(e-?\d+)?)
        """,
        re.VERBOSE
    )

    # 创建一个字典，用于存储每个md_step的数据
    md_forces = defaultdict(list)

    with open(directory, 'r') as file:
        content = file.read()

    for block_match in block_regex.finditer(content):
        md_step = int(block_match.group('md_step'))

        # 将原子数据存储为一个字典的列表，然后按照 ATOMS 的值进行排序
        atom_data = []
        for line_match in line_regex.finditer(block_match.group('block')):
            data = {
                'ATOMS': int(line_match.group('ATOMS')),
                'x': float(line_match.group('x')),
                'y': float(line_match.group('y')),
                'z': float(line_match.group('z')),
                'fx': float(line_match.group('fx')),
                'fy': float(line_match.group('fy')),
                'fz': float(line_match.group('fz')),
            }
            atom_data.append(data)

        # 按照 ATOMS 的值进行排序
        atom_data.sort(key=lambda x: x['ATOMS'])

        # 将排序后的数据添加到对应的md_step中
        md_forces[md_step] = atom_data
    num_md_steps = len(md_forces)
    print(f"There are {num_md_steps} steps in the MD forces data.")

    return md_forces

dft_forces = vasp_forces('./VASP-confirm/')
md_forces = lammps_forces('./lammps/010.dump')

def calculate_total_rmse(dft_forces, md_forces):
    dft_all_forces = []
    md_all_forces = []
    for step in dft_forces.keys():
        if step in md_forces.keys():
            dft_forces_step = np.array([force_dict['force'] for force_dict in dft_forces[step]])
            md_forces_step = np.array([np.array([force_dict['fx'], force_dict['fy'], force_dict['fz']]) for force_dict in md_forces[step]])

            # Assert the shapes of arrays at each step
            assert dft_forces_step.shape == md_forces_step.shape, f"Shapes of arrays are not consistent at step {step}!"

            dft_all_forces.extend(dft_forces_step.flatten())
            md_all_forces.extend(md_forces_step.flatten())

    total_rmse = np.sqrt(mean_squared_error(dft_all_forces, md_all_forces))
    return total_rmse


total_rmse = calculate_total_rmse(dft_forces, md_forces)
print(f"The total RMSE across all steps, atoms and directions is: {total_rmse}")

def plot_force_correlation(dft_forces, md_forces):
    dft_all_forces = []
    md_all_forces = []
    for step in dft_forces.keys():
        if step in md_forces.keys():
            dft_forces_step = np.array([force_dict['force'] for force_dict in dft_forces[step]])
            md_forces_step = np.array([np.array([force_dict['fx'], force_dict['fy'], force_dict['fz']]) for force_dict in md_forces[step]])

            # Assert the shapes of arrays at each step
            assert dft_forces_step.shape == md_forces_step.shape, f"Shapes of arrays are not consistent at step {step}!"

            dft_all_forces.extend(dft_forces_step.flatten())
            md_all_forces.extend(md_forces_step.flatten())

    
    plt.figure(figsize=(10, 10))
    plt.rcParams["font.family"] = "Arial"
    plt.scatter(md_all_forces, dft_all_forces, s=1)
    plt.xlabel('DPMD forces (eV/Angstrom)',fontsize=26)
    plt.ylabel('DFT forces (eV/Angstrom)',fontsize=26)

    lims = [np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()])]  # max of both axes

    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.xlim(lims)
    plt.ylim(lims)

    plt.tick_params(axis='both', labelsize=22)
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(2)

    plt.savefig("force.tiff", dpi=300)
    plt.show()

plot_force_correlation(dft_forces, md_forces)





