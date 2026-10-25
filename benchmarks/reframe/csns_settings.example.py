"""Site configuration matching the current CSNS V100 Slurm environment.

Copy this file to a private ReFrame settings path before running. Set
CSNS_SLURM_ACCOUNT only if your CSNS association requires an explicit account.
"""

import os


access = ["--partition=dggpu", "--qos=normal"]
account = os.environ.get("CSNS_SLURM_ACCOUNT")
if account:
    access.append(f"--account={account}")

site_configuration = {
    "logging": [
        {
            "level": "info",
            "handlers": [
                {
                    "type": "stream",
                    "name": "stdout",
                    "level": "info",
                    "format": "%(message)s",
                }
            ],
            "handlers_perflog": [
                {
                    "type": "filelog",
                    "basedir": "/dg_workfs/LQCD/chensiyang/codex_jobs/lqcdweave-reframe/perflogs",
                    "prefix": "%(check_system)s/%(check_partition)s",
                    "level": "info",
                    "append": True,
                }
            ],
        }
    ],
    "systems": [
        {
            "name": "csns",
            "descr": "CSNS V100 GPU partition",
            "hostnames": [r"ln005"],
            "modules_system": "lmod",
            "prefix": "/dg_workfs/LQCD/chensiyang/codex_jobs/lqcdweave-reframe",
            "partitions": [
                {
                    "name": "dggpu",
                    "scheduler": "slurm",
                    "launcher": "srun",
                    "access": access,
                    "environs": ["python311-cupy"],
                    "resources": [
                        {
                            "name": "gpu",
                            "options": ["--gres=gpu:v100:{num_gpus_per_node}"],
                        }
                    ],
                    "max_jobs": 1,
                }
            ],
        }
    ],
    "environments": [
        {
            "name": "python311-cupy",
            "modules": [
                "lqcd/gpu/mpi/openmpi/5.0.10-ucx-cuda12.4-gcc11",
                "lqcd/gpu/anaconda3/python3.11",
            ],
            "target_systems": ["csns"],
        }
    ],
}
