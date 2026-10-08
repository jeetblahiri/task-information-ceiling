#!/usr/bin/env python3
"""Post-outcome BEM sensitivity of frozen template encoders; no refitting.

Nominal cap, all six tasks/both domains/joint noise. This reuses the existing
coarse/fine nominal gains and source laws, with analytic state-known oracles.
Sample coarse/fine geometry differs; this is not pure mesh convergence.
"""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from tdo_sim.ceiling_benchmark import prepare_transform_experiment

spec=importlib.util.spec_from_file_location('runner',ROOT/'scripts/run_ceiling_benchmark.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
OUT=ROOT/'results/ceiling_benchmark_v1_transfer_resolution'


def main():
    if (OUT/'manifest.json').exists():raise RuntimeError('Diagnostic complete; no overwrite')
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads(runner.CONFIG.read_text());bank=runner.load_banks(config)['bem_sample']
    gainpath=ROOT/'results/ceiling_benchmark_v1_supplements/bem_sample_mesh_nominal_gains.npz'
    decisionpath=ROOT/'results/ceiling_benchmark_v1/anatomy_transfer_decisions.json'
    choices=json.loads(decisionpath.read_text());gains=np.load(gainpath)
    inputs={str(p.relative_to(ROOT)):runner.sha(p) for p in [gainpath,decisionpath,runner.CONFIG,
        ROOT/'resources/ceiling_banks_v1/bem_sample.npz',ROOT/'results/ceiling_benchmark_v1_supplements/geometry_checks.json']}
    rows=[]
    for task in config['tasks']:
        designpath=ROOT/'results/ceiling_benchmark_v1/designs'/f'bem_fsaverage__{task}__joint.npz'
        inputs[str(designpath.relative_to(ROOT))]=runner.sha(designpath)
        design=np.load(designpath)
        for domain in ['check_id','check_shift']:
            selected={policy:next(r['candidate'] for r in choices if r['task']==task and r['domain']==domain and r['policy']==policy)
                for policy in ['development_information','fixed_generic_pca']}
            for mesh in ['ico3','ico4']:
                matched={**bank,'gains_v_per_am':np.broadcast_to(gains[mesh],bank['gains_v_per_am'].shape)}
                state=runner.source_means(matched,task,domain,config)
                experiment=runner.experiment(matched,state,config['noise_regimes']['joint'],19,config)
                key=f'{task}__{domain}__{mesh}'
                payload=dict(plus=experiment.plus,minus=experiment.minus,spatial_covariance=experiment.spatial_covariance,
                    temporal_covariance=experiment.temporal_covariance,background=experiment.background,probabilities=experiment.probabilities)
                row=dict(task=task,domain=domain,mesh=mesh,full_state_known=experiment.oracle_summary(),representations={})
                for policy,name in selected.items():
                    spatial=design[name+'__spatial'];temporal=design[name+'__temporal']
                    encoded=prepare_transform_experiment(experiment,spatial,temporal,name=name)
                    row['representations'][policy]=dict(candidate=name,**encoded.oracle_summary())
                    payload[policy+'__spatial']=spatial;payload[policy+'__temporal']=temporal
                np.savez_compressed(OUT/(key+'.npz'),**payload)
                rows.append(row)
    runner.dump(OUT/'conditions.json',rows)
    runner.dump(OUT/'manifest.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),post_outcome=True,
        scope='24 exact state-known nominal-cap conditions; frozen template encoders; existing sample geometry/model-plus-resolution sensitivity; no new fit or success gate',
        conditions=len(rows),script_sha256=runner.sha(Path(__file__)),input_sha256=inputs,
        dependency_sha256={str(p.relative_to(ROOT)):runner.sha(p) for p in [ROOT/'scripts/run_ceiling_benchmark.py',ROOT/'src/tdo_sim/ceiling_benchmark.py']},
        output_sha256={str(p.relative_to(ROOT)):runner.sha(p) for p in OUT.glob('*') if p.is_file()}))
    for domain in ['check_id','check_shift']:
        print(domain)
        for task in config['tasks']:
            a=[r for r in rows if r['task']==task and r['domain']==domain]
            print(task,[(r['mesh'],round(r['full_state_known']['average_information_bits'],5),
                round(r['representations']['development_information']['average_information_bits'],5),
                round(r['representations']['fixed_generic_pca']['average_information_bits'],5)) for r in a])


if __name__=='__main__':main()
