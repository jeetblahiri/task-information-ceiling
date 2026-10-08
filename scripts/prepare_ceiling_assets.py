#!/usr/bin/env python3
"""Fresh physical operators for the frozen ceiling benchmark; no task scores."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('_MNE_FAKE_HOME_DIR','/tmp/tdo_ceiling_mne')
Path(os.environ['_MNE_FAKE_HOME_DIR']).mkdir(parents=True, exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import mne
from mne.bem import _surfaces_to_bem
from mne.surface import _project_onto_surface
from tdo_sim.geometry import make_nested_geometry, _farthest_order
from tdo_sim.full_study import generate_head

OUT=ROOT/'resources/ceiling_banks_v1'
CONFIG=ROOT/'config/ceiling_benchmark.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def farthest(points,n):
    chosen=[int(np.argmax(points[:,2]))]
    distances=np.sum((points-points[chosen[0]])**2,axis=1)
    while len(chosen)<n:
        distances[chosen]=-1
        k=int(np.argmax(distances));chosen.append(k)
        distances=np.minimum(distances,np.sum((points-points[k])**2,axis=1))
    return np.array(chosen)

def verify_anatomy():
    tree=json.loads(Path('/tmp/tdo_ceiling_testing_tree.json').read_text())
    blobs={x['path']:x for x in tree['tree'] if x['type']=='blob'}
    anatomy=ROOT/'resources/ceiling_anatomy'
    inputs={}
    downloaded=['sample-1280-1280-1280-bem-sol.fif','sample-1280-1280-1280-bem.fif','sample-fiducials.fif','sample-320-320-320-bem-sol.fif']
    for filename in downloaded:
        p=anatomy/'sample/bem'/filename
        data=p.read_bytes(); name='subjects/sample/bem/'+p.name
        digest=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if digest!=blobs[name]['sha']:raise RuntimeError('Git blob mismatch '+name)
        inputs[str(p.relative_to(ROOT))]=sha(p)
    archive=Path('/tmp/tdo_ceiling_MNE-lite-data.tar.gz')
    if sha(archive)!='783c3dd04fe6237e1010c900d2db03fc3f0708bb98f52a708f5ee5277884a94b':
        raise RuntimeError('Official MNE lite checksum mismatch')
    members={
        'MNE-lite-data/MNE-sample-data/subjects/sample/bem/sample-oct-6-src.fif':anatomy/'sample/bem/sample-oct-6-src.fif',
        'MNE-lite-data/MNE-sample-data/MEG/sample/sample_audvis_raw-trans.fif':anatomy/'sample/sample-trans.fif',
    }
    with tarfile.open(archive) as tf:
        for name,p in members.items():
            member=tf.getmember(name)
            if not member.isfile():raise RuntimeError('Required anatomy member is not a file')
            data=tf.extractfile(member).read()
            if p.exists() and p.read_bytes()!=data:raise RuntimeError('Existing anatomy differs')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            inputs[str(p.relative_to(ROOT))]=sha(p)
    dump(anatomy/'provenance.json',{
        'archive_url':'https://osf.io/download/a8qbx',
        'archive_md5':'5f9c4fffed32e79bc2bc2061bf22ce99',
        'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
        'official_config':'MNE1.13.2 datasets/config.py lite_data entry',
        'testing_repository':'https://github.com/mne-tools/mne-testing-data',
        'inspected_git_tree_sha':tree['sha'],
        'git_blob_verification':'SHA1(git blob header + bytes) matched the inspected official tree for every downloaded BEM file',
        'extraction':'Only source space and coordinate transform extracted from lite archive; no EEG/MEG recordings extracted or used',
        'anatomies':'sample individual anatomy; existing fsaverage template remains separate',
        'output_sha256':inputs,
    })
    return anatomy,inputs

def bem_bank(subject,config,anatomy,shared_order):
    base=ROOT/'resources/anatomy/fsaverage' if subject=='fsaverage' else anatomy/'sample'
    src_path=base/'bem/fsaverage-ico-5-src.fif' if subject=='fsaverage' else base/'bem/sample-oct-6-src.fif'
    trans_path=base/'bem/fsaverage-trans.fif' if subject=='fsaverage' else base/'sample-trans.fif'
    trans=mne.read_trans(trans_path,verbose=False)
    if (trans['from'],trans['to'])!=(4,5):raise RuntimeError('Expected head-to-MRI transform')
    source=mne.read_source_spaces(src_path,verbose=False).copy()
    for s in source:
        eligible=s['vertno'][s['rr'][s['vertno'],2]>=.005]
        vert=np.sort(eligible[farthest(s['rr'][eligible],config['bem_sources_per_hemisphere'])])
        s['vertno']=vert;s['inuse']=np.zeros_like(s['inuse']);s['inuse'][vert]=1;s['nuse']=len(vert)
        s['use_tris']=None;s['nuse_tri']=0;s['dist']=None;s['dist_limit']=None
    if subject=='fsaverage':
        model_path=base/'bem/fsaverage-5120-5120-5120-bem.fif'
        original=mne.read_bem_surfaces(model_path,verbose=False)
        model=_surfaces_to_bem(copy.deepcopy(original),[s['id'] for s in original],config['bem_conductivities_s_m'],ico=3,rescale=False)
        bem=mne.make_bem_solution(model,verbose=False)
        finer=mne.read_bem_solution(base/'bem/fsaverage-5120-5120-5120-bem-sol.fif',verbose=False)
        resolution_note='ico3 benchmark versus existing official ico4 sensitivity; not convergence proof'
    else:
        model_path=base/'bem/sample-1280-1280-1280-bem.fif'
        bem=mne.read_bem_solution(base/'bem/sample-1280-1280-1280-bem-sol.fif',verbose=False)
        model=bem['surfs']
        finer=mne.read_bem_solution(base/'bem/sample-320-320-320-bem-sol.fif',verbose=False)
        resolution_note='ico3 benchmark versus official coarser ico2 sensitivity; not convergence proof'
    sigmas=[s['sigma'] for s in model]
    if not np.allclose(sorted(sigmas),sorted(config['bem_conductivities_s_m']),rtol=1e-6,atol=0):raise RuntimeError('BEM conductivity mismatch')
    names=mne.channels.make_standard_montage('biosemi64').ch_names
    info=mne.create_info(names,sfreq=config['sampling_frequency_hz'],ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('fsaverage_1005'),verbose=False)
    starting=np.asarray([ch['loc'][:3] for ch in info['chs']])
    scalp=copy.deepcopy(next(s for s in model if s['id']==4))
    scalp['rr']=mne.transforms.apply_trans(np.linalg.inv(trans['trans']),scalp['rr'])
    def project(points):
        _,_,rr,nn=_project_onto_surface(points,scalp,project_rrs=True,return_nn=True)
        return rr,nn
    sensors,normals=project(starting)
    def forward(positions,solution,allow_resolution_subset=False):
        current=info.copy()
        current.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names,positions)),coord_frame='head'),verbose=False)
        f=mne.make_forward_solution(current,trans=trans,src=source,bem=solution,meg=False,eeg=True,mindist=0,n_jobs=1,verbose=False)
        f=mne.convert_forward_solution(f,surf_ori=True,force_fixed=True,use_cps=False,verbose=False)
        g=np.asarray(f['sol']['data'])
        if (g.shape!=(64,32) and not (allow_resolution_subset and g.shape[0]==64 and g.shape[1]>0)) or not np.isfinite(g).all():raise RuntimeError('Invalid BEM gain')
        return g,f
    nominal,fwd=forward(sensors,bem)
    other,other_fwd=forward(sensors,finer,allow_resolution_subset=True)
    distance=np.linalg.norm(other_fwd['source_rr'][:,None,:]-fwd['source_rr'][None,:,:],axis=2)
    common=np.argmin(distance,axis=1)
    if np.max(distance[np.arange(len(common)),common])>1e-10 or len(set(common.tolist()))!=len(common):raise RuntimeError('Resolution source correspondence invalid')
    car=np.eye(64)-np.ones((64,64))/64
    sensitivity=float(np.linalg.norm(car@(nominal[:,common]-other))/np.linalg.norm(car@nominal[:,common]))
    print(f'{subject}: matched ico3 BEM, sensitivity={sensitivity:.5f}',flush=True)
    seed_base=config['geometry_banks']['bem_'+subject]['seed_base']
    n=sum(config['head_counts'].values());gains=[];params=[];max_surface_error=0.
    for i in range(n):
        seed=seed_base+i*config['head_seed_stride'];rng=np.random.default_rng(seed)
        angle=rng.uniform(-config['bem_cap_rotation_halfwidth_degrees'],config['bem_cap_rotation_halfwidth_degrees'])
        a=np.deg2rad(angle);rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        jitter=rng.uniform(-config['bem_tangential_jitter_halfwidth_m'],config['bem_tangential_jitter_halfwidth_m'],sensors.shape)
        tangent=jitter-np.sum(jitter*normals,axis=1)[:,None]*normals
        pos,_=project(sensors@rot.T+tangent)
        reproj,_=project(pos);err=float(np.max(np.linalg.norm(pos-reproj,axis=1)));max_surface_error=max(max_surface_error,err)
        gain,_=forward(pos,bem);gains.append(gain)
        split='development' if i<8 else 'validation' if i<12 else 'check'
        params.append({'head_id':f'bem_{subject}_{i:03d}','seed':seed,'split':split,'subject':subject,
                       'maximum_contact_displacement_m':float(np.max(np.linalg.norm(pos-sensors,axis=1))),
                       'contact_surface_reprojection_error_m':err,'rotation_degrees':float(angle),'fixed_across_observations':True})
        if (i+1)%12==0:print(f'{subject}: {i+1}/{n} operators',flush=True)
    order=_farthest_order(sensors) if shared_order is None else shared_order
    metadata={'kind':'bem','subject':subject,'anatomy_count':1,'scope':'One anatomy with independent cap perturbations; not human-population draws',
              'bem_ico':3,'triangles_per_layer':[len(s['tris']) for s in model],'conductivities_s_m':sigmas,
              'resolution_sensitivity_relative_car_gain':sensitivity,'resolution_note':resolution_note,
              'resolution_sensitivity_common_sources':len(common),'primary_sources_retained':32,
              'source_orientation_norm_error':float(np.max(np.abs(np.linalg.norm(fwd['source_nn'],axis=1)-1))),
              'contact_surface_reprojection_error_m':max_surface_error,'sensor_names':names,
              'sensor_positions':'Standard named cap projected accurately on anatomical scalp; no measured electrode localization',
              'coordinate_frame':'head','position_units':'m','gain_units':'V/(A m)',
              'montage_order':'fsaverage coverage order shared unchanged across both BEM anatomies'}
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [src_path,trans_path,model_path]}
    return dict(gains_v_per_am=np.stack(gains),nominal_gain_v_per_am=nominal,sensor_positions_m=sensors,
                source_positions_m=fwd['source_rr'],source_orientations=fwd['source_nn'],montage_order=order),params,metadata,inputs,order

def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest_path=OUT/'manifest.json'
    if manifest_path.exists():
        m=json.loads(manifest_path.read_text())
        if m['script_sha256']!=sha(__file__) or m['config_sha256']!=sha(CONFIG):raise RuntimeError('Assets are frozen; script/config changed')
        for name,digest in m['output_sha256'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Asset hash mismatch '+name)
        print('Ceiling assets reused after verification',flush=True);return
    config=json.loads(CONFIG.read_text());start=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    anatomy,inputs=verify_anatomy();params_all={};geometry=make_nested_geometry(n_sources=32,seed=20261008)
    old=json.loads((ROOT/'config/full_study.json').read_text());n=sum(config['head_counts'].values())
    for name,definition in config['geometry_banks'].items():
        if definition['kind']!='sphere':continue
        gains=[];params=[]
        for i in range(n):
            seed=definition['seed_base']+i*config['head_seed_stride']
            gain,p=generate_head(geometry,old['head_perturbations_at_severity1'],definition['severity'],seed)
            p.update(head_id=f'{name}_{i:03d}',split='development' if i<8 else 'validation' if i<12 else 'check',fixed_across_observations=True)
            gains.append(gain);params.append(p)
        # The exact nominal operator is a declared development reference.
        from tdo_sim.forward import make_spherical_forward
        nominal=make_spherical_forward(geometry).gain_v_per_am
        np.savez_compressed(OUT/(name+'.npz'),gains_v_per_am=np.stack(gains),nominal_gain_v_per_am=nominal,
                            sensor_positions_m=geometry.sensor_positions_m,source_positions_m=geometry.source_positions_m,
                            source_orientations=geometry.source_orientations,montage_order=geometry.montages[256])
        dump(OUT/(name+'_parameters.json'),params);params_all[name]=params
        dump(OUT/(name+'_metadata.json'),geometry.metadata())
        print(name+': fresh sphere operators complete',flush=True)
    shared=None
    for subject in ['fsaverage','sample']:
        data,params,metadata,new_inputs,shared=bem_bank(subject,config,anatomy,shared)
        name='bem_'+subject;np.savez_compressed(OUT/(name+'.npz'),**data)
        dump(OUT/(name+'_parameters.json'),params);dump(OUT/(name+'_metadata.json'),metadata)
        params_all[name]=params;inputs.update(new_inputs)
    seeds=[p['seed'] for bank in params_all.values() for p in bank]
    if len(seeds)!=len(set(seeds)):raise RuntimeError('Head seed collision')
    dump(OUT/'manifest.json',{'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
         'elapsed_seconds':time.perf_counter()-start,'script_sha256':sha(__file__),'config_sha256':sha(CONFIG),
         'input_sha256':inputs,'prepared_before_task_outcomes':True,'operators':len(seeds),'unique_head_seeds':len(set(seeds)),
         'anatomy_statement':'Two BEM geometries: one average template and one individual public anatomy; not a sampled human population',
         'output_sha256':{str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('*') if p.is_file()}})
    print(f'Prepared {len(seeds)} new operators in {time.perf_counter()-start:.1f}s',flush=True)

if __name__=='__main__':main()
