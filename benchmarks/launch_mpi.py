"""Launch the QUDA comparison under MPI for Slurm smoke jobs.

The CSNS submit helper runs a Python script directly for one rank. PyQUDA's
MPI initialization needs the same executable to be started by mpirun, even for
one rank, so this thin wrapper re-execs the benchmark under Open MPI.
"""

import os
from pathlib import Path
import sys


def main():
    benchmark = Path(__file__).with_name('compare_quda.py')
    # PyQUDA on CSNS is linked against the site OpenMPI 4.1.6 build while the
    # default module currently exposes OpenMPI 5.0.10.  Use the matching
    # launcher and libraries for this process; the Slurm allocation remains
    # unchanged.
    mpi_root = '/dg_hpc/LQCD/software/x86_gpu/utilities/openmpi-4.1.6-ucx-cuda12.4'
    env = os.environ.copy()
    env['PATH'] = mpi_root + '/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = mpi_root + '/lib:' + env.get('LD_LIBRARY_PATH', '')
    command = [mpi_root + '/bin/mpirun', '-np', '1', sys.executable, '-u',
               str(benchmark), *sys.argv[1:]]
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
