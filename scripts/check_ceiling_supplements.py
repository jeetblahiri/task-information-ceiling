#!/usr/bin/env python3
"""Separate, predeclared mesh and persistent-calibration sensitivity checks."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(key,'1')
os.environ.setdefault('_MNE_FAKE_HOME_DIR','/tmp/tdo_ceiling_mne')
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import mne
from scipy.spatial import cKDTree
from mne.bem import _surfaces_to_bem
from tdo_sim.insight import calibration_information
from tdo_sim.ceiling_benchmark import draw_raw_epochs,summarize_bounded

def import_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

runner=import_script('run_ceiling_benchmark');prep=import_script('prepare_ceiling_assets')
OUT=ROOT/'results/ceiling_benchmark_v1_supplements'
SPEC=ROOT/'config/ceiling_supplement_checks.json'

def read_triangle_surface(path):
    """Strict standard FreeSurfer triangle geometry; coordinates are mm.

    This small binary reader avoids installing an optional imaging package.
    It ignores trailing volume metadata, which is not used for BEM geometry.
    Native geometry is verified against official Git blobs and coarse FIF
    surface vertices before constructing a finer conductor model.
    """
    with Path(path).open('rb') as f:
        if int.from_bytes(f.read(3),'big')!=16777214:raise RuntimeError('Not a triangle surface')
        f.readline();f.readline()
        nv,nt=struct.unpack('>ii',f.read(8))
        if not 3<nv<1000000 or not 3<nt<2000000:raise RuntimeError('Invalid surface counts')
        rr=np.frombuffer(f.read(nv*12),dtype='>f4').astype(float).reshape(nv,3)/1000
        tris=np.frombuffer(f.read(nt*12),dtype='>i4').astype(int).reshape(nt,3)
    if not np.isfinite(rr).all() or tris.min()<0 or tris.max()>=nv:raise RuntimeError('Invalid surface geometry')
    return {'rr':rr,'tris':tris,'np':nv,'ntri':nt,'coord_frame':5}

def fine_nominal(subject,bank,config):
    base=ROOT/'resources/anatomy/fsaverage' if subject=='fsaverage' else ROOT/'resources/ceiling_anatomy/sample'
    src_path=base/'bem/fsaverage-ico-5-src.fif' if subject=='fsaverage' else base/'bem/sample-oct-6-src.fif'
    trans_path=base/'bem/fsaverage-trans.fif' if subject=='fsaverage' else base/'sample-trans.fif'
    if subject=='fsaverage':
        bem=mne.read_bem_solution(base/'bem/fsaverage-5120-5120-5120-bem-sol.fif',verbose=False)
        geometry_audit={'source':'official ico4 solution already checksum-verified','triangles_per_layer':[len(s['tris']) for s in bem['surfs']]}
    else:
        tree=json.loads(Path('/tmp/tdo_ceiling_testing_tree.json').read_text())
        blob={x['path']:x for x in tree['tree'] if x['type']=='blob'}
        surfaces=[];checks={}
        for name,identity in [('outer_skin',4),('outer_skull',3),('inner_skull',1)]:
            p=base/'bem_native'/f'{name}.surf';data=p.read_bytes()
            gitsha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if gitsha!=blob[f'subjects/sample/bem/{name}.surf']['sha']:raise RuntimeError('Native surface Git blob mismatch')
            s=read_triangle_surface(p);s['id']=identity;surfaces.append(s);checks[name]={'vertices':len(s['rr']),'triangles':len(s['tris']),'sha256':runner.sha(p),'git_blob_sha1':gitsha}
        # Validate the binary parser against an independent official ASCII
        # representation. Official coarse FIF may include different smoothing;
        # that discrepancy must be reported as model choice, not pure mesh.
        ascii_discrepancies=[]
        for name in ['outer_skin','outer_skull','inner_skull']:
            p=base/'bem_native'/f'{name}.tri';data=p.read_bytes()
            gitsha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if gitsha!=blob[f'subjects/sample/bem/{name}.tri']['sha']:raise RuntimeError('ASCII surface Git blob mismatch')
            with p.open() as f:
                nv=int(f.readline());rows=np.array([[float(v) for v in f.readline().split()[1:]] for _ in range(nv)])/1000
            s=surfaces[['outer_skin','outer_skull','inner_skull'].index(name)]
            if rows.shape!=s['rr'].shape:raise RuntimeError('ASCII/binary surface count mismatch')
            ascii_discrepancies.append(float(np.max(np.abs(rows-s['rr']))))
        if max(ascii_discrepancies)>1e-7:raise RuntimeError('Native surface parser or unit mismatch')
        official=mne.read_bem_surfaces(base/'bem/sample-1280-1280-1280-bem.fif',verbose=False)
        discrepancies=[]
        for s in surfaces:
            old=next(t for t in official if t['id']==s['id'])
            errors,_=cKDTree(s['rr']).query(old['rr'])
            discrepancies.append(float(errors.max()))
        model=_surfaces_to_bem(surfaces,[s['id'] for s in surfaces],config['bem_conductivities_s_m'],ico=4,rescale=False)
        cache=ROOT/'resources/ceiling_anatomy/sample/bem/sample-derived-5120-5120-5120-bem-sol.fif'
        if cache.exists():
            cache_record=json.loads(cache.with_suffix('.json').read_text())
            if runner.sha(cache)!=cache_record['sha256']:raise RuntimeError('Derived ico4 cache changed')
            bem=mne.read_bem_solution(cache,verbose=False)
        else:
            print('Computing sample ico4 BEM solution on verified native surfaces',flush=True)
            bem=mne.make_bem_solution(model,verbose=False)
            mne.write_bem_solution(cache,bem,overwrite=False,verbose=False)
            runner.dump(cache.with_suffix('.json'),{'sha256':runner.sha(cache),'surface_checks':checks,
                'mne_version':mne.__version__,'conductivities_s_m':config['bem_conductivities_s_m']})
        geometry_audit={'native_surfaces':checks,'max_binary_ascii_surface_coordinate_difference_m':max(ascii_discrepancies),
                        'max_official_coarse_to_native_nearest_vertex_distance_m':max(discrepancies),
                        'sensitivity_scope':'Official coarse model versus native-surface finer model includes geometry/smoothing differences; not pure mesh convergence',
                        'triangles_per_layer':[len(s['tris']) for s in bem['surfs']],'derived_solution_sha256':runner.sha(cache)}
    source=mne.read_source_spaces(src_path,verbose=False).copy()
    for s in source:
        eligible=s['vertno'][s['rr'][s['vertno'],2]>=.005]
        vert=np.sort(eligible[prep.farthest(s['rr'][eligible],config['bem_sources_per_hemisphere'])])
        s['vertno']=vert;s['inuse']=np.zeros_like(s['inuse']);s['inuse'][vert]=1;s['nuse']=len(vert)
        s['use_tris']=None;s['nuse_tri']=0;s['dist']=None;s['dist_limit']=None
    names=bank['metadata']['sensor_names'];info=mne.create_info(names,sfreq=config['sampling_frequency_hz'],ch_types='eeg')
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names,bank['sensor_positions_m'])),coord_frame='head'),verbose=False)
    fwd=mne.make_forward_solution(info,trans=mne.read_trans(trans_path,verbose=False),src=source,bem=bem,
        meg=False,eeg=True,mindist=0,n_jobs=1,verbose=False)
    fwd=mne.convert_forward_solution(fwd,surf_ori=True,force_fixed=True,use_cps=False,verbose=False)
    gain=fwd['sol']['data']
    if gain.shape!=(64,32):raise RuntimeError('Fine-resolution source count changed')
    if not np.allclose(fwd['source_rr'],bank['source_positions_m'],rtol=0,atol=1e-9):raise RuntimeError('Fine-resolution source positions changed')
    if not np.allclose(fwd['source_nn'],bank['source_orientations'],rtol=0,atol=1e-6):raise RuntimeError('Fine-resolution source orientations changed')
    return gain,geometry_audit

def mesh_checks(banks,config,spec):
    rows=[];audits={}
    for subject in spec['mesh']['anatomies']:
        name='bem_'+subject;bank=banks[name];finer,audit=fine_nominal(subject,bank,config);audits[name]=audit
        q=runner.car_basis(64).T
        relative=float(np.linalg.norm(q@(finer-bank['nominal_gain_v_per_am']))/np.linalg.norm(q@finer))
        audits[name]['ico3_to_ico4_relative_car_gain']=relative
        np.savez_compressed(OUT/f'{name}_mesh_nominal_gains.npz',ico3=bank['nominal_gain_v_per_am'],ico4=finer)
        coarse_bank={**bank,'gains_v_per_am':np.broadcast_to(bank['nominal_gain_v_per_am'],bank['gains_v_per_am'].shape)}
        fine_bank={**bank,'gains_v_per_am':np.broadcast_to(finer,bank['gains_v_per_am'].shape)}
        for task in config['tasks']:
            low_state=runner.source_means(coarse_bank,task,'check_id',config)
            high_state=runner.source_means(fine_bank,task,'check_id',config)
            for noise,regime in config['noise_regimes'].items():
                low=runner.experiment(coarse_bank,low_state,regime,19,config).oracle_summary()
                high=runner.experiment(fine_bank,high_state,regime,19,config).oracle_summary()
                difference=np.array(high['information_bits'])-np.array(low['information_bits'])
                rows.append({'bank':name,'task':task,'noise':noise,'channels':19,
                    'ico3_known_state_bits':runner.info_bits(low),'ico4_known_state_bits':runner.info_bits(high),
                    'difference_bits':float(difference.mean()),'maximum_absolute_source_state_difference_bits':float(np.max(np.abs(difference))),
                    'scope':'Matched source/static/epoch-background states on nominal cap; model/resolution sensitivity (sample includes supplied geometry differences), not physiological accuracy'})
        print(f'Mesh sensitivity complete {name}; gain difference={relative:.5f}',flush=True)
    return rows,audits

def operational_calibration(dev,check,n_calibration,n_subjects,test_epochs,random_seed):
    rng=np.random.default_rng(random_seed);heads=rng.choice(check.n_heads,n_subjects,p=check.probabilities)
    labels=rng.integers(0,2,(n_subjects,n_calibration))*2-1
    if n_calibration:
        draw=draw_raw_epochs(check,n_subjects*n_calibration,random_seed+1,head_indices=np.repeat(heads,n_calibration),labels=labels.ravel())
        coordinates=dev.likelihood_coordinates(dev.whiten(draw.epochs)).reshape(n_subjects,n_calibration,-1)
        weights=dev.channel.posterior_heads(coordinates,labels)
    else:weights=np.broadcast_to(dev.probabilities,(n_subjects,dev.n_heads))
    test=draw_raw_epochs(check,n_subjects*test_epochs,random_seed+2,head_indices=np.repeat(heads,test_epochs))
    coordinates=dev.likelihood_coordinates(dev.whiten(test.epochs))
    probabilities=dev.channel.posterior(coordinates,np.repeat(weights,test_epochs,axis=0))
    p=np.clip(probabilities,1e-6,1-1e-6);positive=(test.labels+1)/2
    error=(np.where(p>=.5,1,-1)!=test.labels).reshape(n_subjects,test_epochs).mean(axis=1)
    loss=-(positive*np.log2(p)+(1-positive)*np.log2(1-p)).reshape(n_subjects,test_epochs).mean(axis=1)
    return {'error':summarize_bounded(error,(0.,1.)),'log_loss_bits':summarize_bounded(loss,(0.,math_logloss_max())),
        'subject_indices':heads.tolist(),'subject_errors':error.tolist(),'subject_log_losses_bits':loss.tolist(),
        'scope':'Development-only simulator-revealed prototype prior on truly unseen states; predictive risk/log loss, not true MI',
        'n_subjects':n_subjects,'test_epochs_per_subject':test_epochs,'probability_clip':1e-6}

def math_logloss_max():return float(-np.log2(1e-6))

def calibration_checks(banks,config,spec):
    rows=[];operational=[];settings=spec['calibration']
    regime=config['noise_regimes'][settings['noise_regime']]
    for bank_name,bank in banks.items():
        for task in settings['tasks']:
            dev=runner.experiment(bank,runner.source_means(bank,task,'development',config),regime,19,config)
            for domain in settings['domains']:
                check=runner.experiment(bank,runner.source_means(bank,task,domain,config),regime,19,config)
                for n in settings['known_label_epoch_budgets']:
                    random_seed=runner.seed(config,spec['seed_salt'],'calibration',bank_name,task,domain,n)
                    result=calibration_information(check.channel,n,n_samples=settings['exact_information_outer_draws'],seed=random_seed)
                    rows.append({'bank':bank_name,'task':task,'domain':domain,'noise':settings['noise_regime'],'calibration_epochs':n,**result,
                        'scope':'Exact check-bank prior conditional information; all calibration and test epochs share one state; finite-prior oracle knows latent prototypes'})
                    predicted=operational_calibration(dev,check,n,settings['operational_subject_draws'],settings['operational_test_epochs_per_subject'],random_seed+3)
                    operational.append({'bank':bank_name,'task':task,'domain':domain,'calibration_epochs':n,**predicted})
                print('Calibration sensitivity complete '+bank_name+'/'+task+'/'+domain,flush=True)
    return rows,operational

def main():
    if (OUT/'manifest.json').exists():raise RuntimeError('Supplement completed; do not overwrite')
    OUT.mkdir(parents=True,exist_ok=True);config=json.loads(runner.CONFIG.read_text());spec=json.loads(SPEC.read_text())
    if runner.sha(runner.CONFIG)!=spec['main_config_sha256']:raise RuntimeError('Main config mismatch')
    source_hashes={str(p.relative_to(ROOT)):runner.sha(p) for p in [Path(__file__),SPEC,ROOT/'scripts/run_ceiling_benchmark.py',ROOT/'scripts/prepare_ceiling_assets.py']}
    runner.dump(OUT/'pre_outcome_freeze.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source_hashes,
        'predeclaration_record_sha256':runner.sha(ROOT/'evidence/ceiling_supplement_pre_outcome_freeze.json')})
    start=time.perf_counter();banks=runner.load_banks(config)
    mesh,audits=mesh_checks(banks,config,spec);calibration,operational=calibration_checks(banks,config,spec)
    for name,value in [('mesh_sensitivity',mesh),('geometry_checks',audits),('calibration_information',calibration),('calibration_operational',operational)]:runner.dump(OUT/(name+'.json'),value)
    for name,digest in source_hashes.items():
        if runner.sha(ROOT/name)!=digest:raise RuntimeError('Supplement source changed during run')
    runner.dump(OUT/'manifest.json',{'protocol':spec['protocol'],'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds':time.perf_counter()-start,'source_sha256':source_hashes,'main_config_sha256':runner.sha(runner.CONFIG),
        'counts':{'mesh_conditions':len(mesh),'calibration_information_conditions':len(calibration),'operational_conditions':len(operational)},
        'output_sha256':{str(p.relative_to(ROOT)):runner.sha(p) for p in OUT.glob('*') if p.is_file()}})
    print('Supplement checks completed',flush=True)

if __name__=='__main__':main()
