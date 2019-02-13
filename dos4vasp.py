import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math

np.random.seed(1)

def get_nbands_nele(prefix=""):
    input_file = open("vasprun.xml", "r")

    nbands = 0
    nelect = 0
    for line in input_file:
        if 'name="NBANDS"' in line:
            nbands = int(line.split()[-1][:-4])
        if 'name="NELECT"' in line:
            nelect = int(float(line.split()[-1][:-4]))
            break

    input_file.close()
    return nbands, nelect


def get_eigenvalue(prefix=""):
    input_file = open("vasprun.xml", "r")
    eigen_values = []

    tag = False
    start_tag = '<eigenvalues>'

    # print(prefix)
    for line in input_file:
        if line.strip() == start_tag:
            tag = True
        elif tag:
            if line.strip() == "</eigenvalues>":
                tag = False
                break
            data_list = line.split()
            if data_list[0] == '<r>':
                data_list = [float(element) for element in data_list[1:-1]]
                eigen_values.append(data_list)

    input_file.close()
    return eigen_values


def get_vbm(prefix=""):
    eigenvalues = get_eigenvalue(prefix=prefix)
    nbands, nele = get_nbands_nele(prefix=prefix)

    num_kpoint = int(len(eigenvalues)/nbands)
    vbm_position = -math.inf
    vbm_kpoint = 0

    for kpoint in range(0, num_kpoint):
        eigenvalue = eigenvalues[kpoint*nbands + nele//2 - 1][0]
        if vbm_position < eigenvalue:
            vbm_position = eigenvalue
            vbm_kpoint = kpoint + 1

    # print('VBM poistion:', vbm_position, "; VBM Kpoints index:", vbm_kpoint)
    return vbm_position




def ion_name(prefix=""):
    input_file = open("vasprun.xml", "r")
    ions = []

    tag = False
    start_tag = '<field type="int">atomtype</field>'
    for line in input_file:
        if line.strip() == start_tag:
            tag = True
        elif tag:
            if line.strip() == "</set>":
                tag = False
                break
            data_list = line.strip()[7:9].strip()
            if data_list != '':
                ions.append(data_list)

    input_file.close()

    num_ndos = 3000
    ion_index = []
    for ion in range(0, len(ions)):
        for _ in range(0, num_ndos):
            ion_index.append(ions[ion])
    return np.asarray(ion_index)


def grep_dos(prefix=""):
    input_file = open("vasprun.xml", "r")
    outfile_total = open("total_dos.csv", "w")
    outfile_partial = open("partial_dos.csv", "w")
    tag = False
    reading_total_dos = False
    reading_partial_dos = False
    start_tag = '<set comment="spin 1">'

    column_names_total = "energy,total,integrate\n"
    column_names_partial = "energy,s,px,py,pz,dxy,dxz,dzy,dx2,dx-y\n"
    outfile_total.write(column_names_total)
    outfile_partial.write(column_names_partial)


    for line in input_file:

        curr_line = line.strip()
        if curr_line == '<total>':
            reading_total_dos = True
        if curr_line == '</total>':
            reading_total_dos = False
        if curr_line == '<partial>':
            reading_partial_dos = True
        if curr_line == '</partial>':
            reading_partial_dos = False

        if tag and reading_total_dos:
            if line.strip() == "</set>":
                tag = False
                reading_total_dos = False
            data_list = line.split()[1:-1]
            outfile_total.write(','.join(data_list)+'\n')

        if tag and reading_partial_dos:
            if curr_line == '</partial>':
                if line.strip() == "</set>":
                    tag = False

            data_list = line.split()[1:-1]
            outfile_partial.write(','.join(data_list)+'\n')

        if start_tag in curr_line and (reading_total_dos or reading_partial_dos):
            tag = True


    input_file.close()
    outfile_total.close()
    outfile_partial.close()



def cal_pdos_orbital(prefix=""):
    dos_df = pd.read_csv("partial_dos.csv")
    pdos_orbital_df = dos_df[np.logical_and((dos_df.energy != 'comment="ion'), (dos_df.energy != 'comment="spin'))]
    s_orbital = pdos_orbital_df.loc[:]['s']
    p_orbital = pdos_orbital_df.loc[:]['px'] + pdos_orbital_df.loc[:]['py'] + pdos_orbital_df.loc[:]['pz']
    d_orbital = pdos_orbital_df.loc[:]['dxy'] + pdos_orbital_df.loc[:]['dzy'] + pdos_orbital_df.loc[:]['dxz'] \
                + pdos_orbital_df.loc[:]['dx2'] + pdos_orbital_df.loc[:]['dx-y']

    pdos_orbital = s_orbital.to_frame(name='s')
    pdos_orbital['p'], pdos_orbital['d'] = p_orbital, d_orbital


    energy_ev = pdos_orbital_df.loc[:]['energy']
    pdos_orbital.insert(0, 'energy', energy_ev)
    pdos_orbital.insert(0, 'ion_index', ion_name(prefix=prefix))
    pdos_orbital.to_csv('pdos_orbital.csv', index_label='index')

    pdos = pdos_orbital.groupby(['ion_index', 'energy']).sum().sum(level=['ion_index', 'energy']).reset_index()
    pdos.energy = pdos.energy.astype(float)
    pdos = pdos.sort_values(by=['energy'], ascending=True)
    pdos.to_csv('pdos.csv', index_label='index')


def plot_dos(prefix="", ax=None):

    dos_total = pd.read_csv('total_dos.csv')

    # vbm_position = get_vbm(prefix=prefix)
    dos_total['vbm_position'] = get_vbm(prefix=prefix)
    dos_total['vbm_shift'] = dos_total['energy']-dos_total['vbm_position']


    dos_partial = pd.read_csv('pdos.csv', index_col=1)
    dos_total['Zr_d'] = dos_partial.loc['Zr']['d'].values
    dos_total['S_p'] = dos_partial.loc['S']['p'].values

    # ion_orbital = dos_partial.loc['Zr']
    dos_total.plot(kind='line', linewidth=1.0, x='vbm_shift', y=['total', 'Zr_d', 'S_p'], ax=ax)

    plt.xlabel('Energy(eV)',fontsize=12)
    plt.ylabel('Density of States',fontsize=12)

    plt.xlim((-10, 12))
    plt.ylim((0, 100))
    plt.title('DOS', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # plt.subplots_adjust(top=0.96, bottom=0.08, left=0.10, right=0.95, hspace=0.75,
    #                     wspace=0.25)

def main():
    # fig = plt.figure(figsize=(15,8))
    phases = ['vasprun.xml']

    for phase in phases:
        grep_dos(prefix=phase)
        cal_pdos_orbital(prefix=phase)
        plot_dos(prefix=phase)

    plt.show()


if __name__ == "__main__":
    main()

