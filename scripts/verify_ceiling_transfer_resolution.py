#!/usr/bin/env python3
"""Independent oracle/physical-map audit of post-outcome transfer sensitivity.

Imports no study implementation. The separate main verifier provides independently
implemented NumPy/SciPy reconstruction, Cholesky, Sherman--Morrison and GH routines.
All scientific results, representations, and parameters remain read-only.
"""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ceiling_benchmark_v1_transfer_resolution'
EVIDENCE=ROOT/'evidence/ceiling_transfer_resolution_integrity_checks.json'
spec=importlib.util.spec_from_file_location('independent_main_audit',ROOT/'scripts/verify_ceiling_benchmark_outputs.py')
independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)


def main():
    start=time.perf_counter();audit=independent.Audit()
    read,sha,npz=independent.read,independent.sha,independent.npz
    manifest=read(OUT/'manifest.json');config=read(ROOT/'config/ceiling_benchmark.json')
    audit.mapping('sensitivity_inputs',manifest['input_sha256'])
    audit.mapping('sensitivity_dependencies',manifest['dependency_sha256'])
    audit.mapping('sensitivity_outputs',manifest['output_sha256'])
    audit.require(manifest['script_sha256']==sha(ROOT/'scripts/diagnose_ceiling_transfer_resolution.py'),'diagnostic_script_identity')
    audit.require(manifest['post_outcome'] and 'no new fit or success gate' in manifest['scope'],'post_outcome_scope')
    main_manifest=read(ROOT/'results/ceiling_benchmark_v1/run_manifest.json')
    supplement_manifest=read(ROOT/'results/ceiling_benchmark_v1_supplements/manifest.json')
    audit.mapping('unchanged_main_outputs',main_manifest['output_sha256'])
    audit.mapping('unchanged_main_sources',main_manifest['source_sha256_after'])
    audit.mapping('unchanged_supplement_outputs',supplement_manifest['output_sha256'])
    audit.mapping('unchanged_supplement_sources',supplement_manifest['source_sha256'])
    rows=read(OUT/'conditions.json')
    expected_keys={(task,domain,mesh) for task in config['tasks'] for domain in ['check_id','check_shift'] for mesh in ['ico3','ico4']}
    audit.require(len(rows)==manifest['conditions']==24 and {(r['task'],r['domain'],r['mesh']) for r in rows}==expected_keys,'complete_frozen_sensitivity_grid')
    bank=npz(ROOT/'resources/ceiling_banks_v1/bem_sample.npz')
    bank['parameters']=read(ROOT/'resources/ceiling_banks_v1/bem_sample_parameters.json')
    gains=npz(ROOT/'results/ceiling_benchmark_v1_supplements/bem_sample_mesh_nominal_gains.npz')
    audit.close('coarse_nominal_gain_identity',gains['ico3'],bank['nominal_gain_v_per_am'],atol=0.,rtol=0.)
    choices=read(ROOT/'results/ceiling_benchmark_v1/anatomy_transfer_decisions.json')
    metadata=read(ROOT/'results/ceiling_benchmark_v1/case_metadata.json')
    states={(r['bank'],r['task'],r['domain']):r['states'] for r in metadata}
    additional_inputs={'results/ceiling_benchmark_v1/case_metadata.json':sha(ROOT/'results/ceiling_benchmark_v1/case_metadata.json')}
    numerical=[];dense_maximum=0.;summaries=[]
    for row in rows:
        task,domain,mesh=row['task'],row['domain'],row['mesh'];key='__'.join((task,domain,mesh))
        saved=npz(OUT/(key+'.npz'))
        current=dict(bank);current['gains_v_per_am']=np.broadcast_to(gains[mesh],bank['gains_v_per_am'].shape)
        parameters=states['bem_sample',task,domain]
        audit.require(len(parameters)==24 and len({p['state_seed'] for p in parameters})==24,'source_states_fixed:'+key)
        source=independent.reconstruct_source(current,task,parameters,config)
        physical=independent.physical_law(current,source,19,config['noise_regimes']['joint'],config)
        for field in ['plus','minus','background','spatial_covariance','temporal_covariance']:
            audit.close('raw_physical_law:'+key+':'+field,saved[field],physical[field],atol=1e-20,rtol=2e-12)
        audit.close('fixed_uniform_prior:'+key,saved['probabilities'],np.full(24,1/24),atol=0.,rtol=0.)
        full=independent.oracle_law(physical)
        independent.oracle_check(audit,key+'full',full,row['full_state_known'])
        design_path=ROOT/'results/ceiling_benchmark_v1/designs'/f'bem_fsaverage__{task}__joint.npz'
        design=npz(design_path)
        summary={'task':task,'domain':domain,'mesh':mesh,'full_state_known_bits':float(full['information_bits'].mean())}
        for policy in ['development_information','fixed_generic_pca']:
            choice=next(r['candidate'] for r in choices if r['task']==task and r['domain']==domain and r['policy']==policy)
            representation=row['representations'][policy]
            audit.require(representation['candidate']==choice,'frozen_template_choice:'+key+policy)
            rs,tb=saved[policy+'__spatial'],saved[policy+'__temporal']
            audit.require(np.array_equal(rs,design[choice+'__spatial']) and np.array_equal(tb,design[choice+'__temporal']),
                'byte_exact_frozen_encoder:'+key+policy)
            audit.require(rs.shape[1]==18 and tb.shape[1]==64 and len(rs)*len(tb)==64 and np.linalg.matrix_rank(rs)==len(rs)
                and np.linalg.matrix_rank(tb)==len(tb),'physical_feature_budget_and_rank:'+key+policy)
            audit.close('temporal_average_unit_weights:'+key+policy,tb.sum(axis=1),np.ones(len(tb)),atol=0.,rtol=0.)
            supports=[np.flatnonzero(t) for t in tb]
            audit.require(all(len(s)>0 and np.array_equal(s,np.arange(s[0],s[-1]+1)) for s in supports)
                and np.array_equal(np.concatenate(supports),np.arange(64)) and all(np.all(tb[i,s]==1/len(s)) for i,s in enumerate(supports)),
                'contiguous_raw_time_averages:'+key+policy)
            encoded=independent.transform(physical,rs,tb);reference=independent.oracle_law(encoded)
            independent.oracle_check(audit,key+policy,reference,representation)
            # Independent dense64-D solve, avoiding whitening and rank-one
            # identity entirely, for every represented state in all48 encoders.
            base=np.kron(encoded['spatial_covariance'],encoded['temporal_covariance'])
            dense=[]
            for k in range(24):
                delta=(encoded['plus'][k]-encoded['minus'][k]).reshape(-1)
                b=encoded['background'][k].reshape(-1)
                cov=base+np.outer(b,b)
                dense.append(np.sqrt(max(0.,float(delta@np.linalg.solve(cov,delta)))))
            dense=np.asarray(dense)
            dense_maximum=max(dense_maximum,float(np.max(np.abs(dense-reference['separations']))))
            audit.close('dense64D_covariance_oracle:'+key+policy,reference['separations'],dense,atol=2e-10)
            if np.any(reference['separations']>full['separations']+1e-10):
                numerical.append({'case':key,'policy':policy,'check':'state_known_separation_DPI'})
            if np.any(reference['information_bits']>full['information_bits']+1e-10):
                numerical.append({'case':key,'policy':policy,'check':'state_known_MI_DPI'})
            if np.any(reference['bayes_errors']<full['bayes_errors']-1e-10):
                numerical.append({'case':key,'policy':policy,'check':'state_known_Bayes_risk_DPI'})
            summary[policy+'_bits']=float(reference['information_bits'].mean())
            summary[policy+'_bayes_error']=float(reference['bayes_errors'].mean())
            audit.counts['frozen_template_encoders_checked']+=1
            audit.counts['dense64D_state_oracles']+=24
        summaries.append(summary);audit.counts['sensitivity_raw_laws']=audit.counts['sensitivity_raw_laws']+1
    for task in config['tasks']:
        for domain in ['check_id','check_shift']:
            coarse=npz(OUT/(task+'__'+domain+'__ico3.npz'));fine=npz(OUT/(task+'__'+domain+'__ico4.npz'))
            for field in ['spatial_covariance','temporal_covariance','probabilities','development_information__spatial',
                          'development_information__temporal','fixed_generic_pca__spatial','fixed_generic_pca__temporal']:
                audit.require(np.array_equal(coarse[field],fine[field]),'coarse_fine_non_gain_identity:'+task+domain+field)
    output={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,
        'passed':not audit.failures and not numerical,'failures':audit.failures,'numerical_flags':numerical,
        'hash_checks':audit.hashes,'maximum_absolute_residuals':audit.maximum_residual,'recomputed_counts':dict(audit.counts),
        'dense64D_separation_maximum_absolute_residual':dense_maximum,'independently_recomputed_results':summaries,
        'scope':'Post-outcome nominal-cap sensitivity with all6tasks and2domains, same frozen template encoders and source states; conditional analytic state-known oracles only.',
        'limitations':['The sample fine/coarse conductor surfaces differ geometrically; this is model-plus-resolution sensitivity, not pure mesh convergence or validation against physiology.',
            '24source states share the nominal cap operator; these are neither24independent anatomies nor a new hidden-mixture MI study.',
            'Exact binary Gaussian state-known MI uses independent256-point Gauss-Hermite integration; uncertainty flags here concern floating-point oracle DPI, not Monte Carlo intervals.',
            'Main source-coordinate/normal identity and finer forward runtime assertions were independently audited in the separate supplement check; no costly finer forward regeneration was performed.'],
        'audit_source_sha256':sha(__file__),'independent_calculation_source_sha256':sha(ROOT/'scripts/verify_ceiling_benchmark_outputs.py'),
        'additional_input_sha256':additional_inputs,'sensitivity_manifest_sha256':sha(OUT/'manifest.json'),
        'interpreter':sys.executable,'packages':{'numpy':np.__version__}}
    EVIDENCE.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':output['passed'],'failures':len(audit.failures),'flags':len(numerical),'seconds':output['elapsed_seconds'],'evidence':str(EVIDENCE)},indent=2))
    if not output['passed']:sys.exit(1)


if __name__=='__main__':main()
