#!/usr/bin/env python3
"""Independent immutable-supplement audit; no fits or scientific reruns."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import time
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('_MNE_FAKE_HOME_DIR', '/tmp/tdo_ceiling_supplement_audit_mne')
Path(os.environ['_MNE_FAKE_HOME_DIR']).mkdir(parents=True, exist_ok=True)
import numpy as np
import mne
from scipy.linalg import helmert
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/ceiling_benchmark_v1_supplements'
EVIDENCE = ROOT / 'evidence/ceiling_supplement_integrity_checks.json'
MAIN_MANIFEST_SHA = 'c96ce7245b9e25844c61f98a02ad6d72712f9229400e6d18c7885c56688a42ba'

# Reuse ONLY independently authored NumPy/SciPy audit calculations. This
# imports neither the scientific runner nor tdo_sim's likelihood utilities.
spec = importlib.util.spec_from_file_location('independent_ceiling_audit', ROOT / 'scripts/verify_ceiling_benchmark_outputs.py')
independent = importlib.util.module_from_spec(spec); spec.loader.exec_module(independent)


def binary_surface(path):
    with Path(path).open('rb') as stream:
        magic = int.from_bytes(stream.read(3), 'big')
        if magic != 16777214: raise ValueError('Expected FreeSurfer triangle surface')
        stream.readline(); stream.readline()
        vertices, triangles = struct.unpack('>ii', stream.read(8))
        position = np.frombuffer(stream.read(12 * vertices), dtype='>f4').reshape(vertices, 3).astype(float) * .001
        faces = np.frombuffer(stream.read(12 * triangles), dtype='>i4').reshape(triangles, 3)
    return position, faces


def official_source_identity(subject):
    base = ROOT / ('resources/anatomy/fsaverage' if subject == 'fsaverage' else 'resources/ceiling_anatomy/sample')
    source_path = base / ('bem/fsaverage-ico-5-src.fif' if subject == 'fsaverage' else 'bem/sample-oct-6-src.fif')
    transform_path = base / ('bem/fsaverage-trans.fif' if subject == 'fsaverage' else 'sample-trans.fif')
    sources = mne.read_source_spaces(source_path, verbose=False)
    transform = mne.read_trans(transform_path, verbose=False)
    if (transform['from'], transform['to']) != (4, 5): raise ValueError('Expected head-to-MRI transform')
    head = np.linalg.inv(transform['trans'])
    positions, normals = [], []
    for source in sources:
        eligible = source['vertno'][source['rr'][source['vertno'], 2] >= .005]
        points = source['rr'][eligible]
        selected = [int(np.argmax(points[:, 2]))]
        minimum = np.sum((points-points[selected[0]])**2, axis=1)
        while len(selected) < 16:
            minimum[selected] = -1
            next_index = int(np.argmax(minimum)); selected.append(next_index)
            minimum = np.minimum(minimum, np.sum((points-points[next_index])**2, axis=1))
        vertices = np.sort(eligible[selected])
        positions.append(source['rr'][vertices] @ head[:3, :3].T + head[:3, 3])
        normals.append(source['nn'][vertices] @ head[:3, :3].T)
    return np.vstack(positions), np.vstack(normals), source_path, transform_path


def check_information_summary(audit, key, saved, width, low, high, n):
    half = width * np.sqrt(np.log(40) / (2*n))
    if 'bits' in saved:
        mean, lower_key, upper_key = saved['bits'], 'confidence_lower_bits', 'confidence_upper_bits'
    else:
        mean, lower_key, upper_key = saved['estimate'], 'confidence_lower', 'confidence_upper'
    audit.close('exact_summary_concentration:' + key + lower_key, saved[lower_key], max(low, mean-half))
    audit.close('exact_summary_concentration:' + key + upper_key, saved[upper_key], min(high, mean+half))
    audit.require(np.isfinite(mean) and low <= mean <= high, 'exact_summary_mean_range:' + key)


def main():
    started = time.perf_counter(); audit = independent.Audit()
    read, sha, npz = independent.read, independent.sha, independent.npz
    manifest, freeze = read(OUT/'manifest.json'), read(OUT/'pre_outcome_freeze.json')
    config, supplement = read(ROOT/'config/ceiling_benchmark.json'), read(ROOT/'config/ceiling_supplement_checks.json')
    main_manifest = read(ROOT/'results/ceiling_benchmark_v1/run_manifest.json')
    audit.require(sha(ROOT/'results/ceiling_benchmark_v1/run_manifest.json') == MAIN_MANIFEST_SHA, 'main_manifest_preserved')
    audit.mapping('main_sources', main_manifest['source_sha256_after'])
    audit.mapping('main_outputs', main_manifest['output_sha256'])
    audit.mapping('supplement_sources', manifest['source_sha256'])
    audit.mapping('supplement_outputs', manifest['output_sha256'])
    audit.require(freeze['source_sha256'] == manifest['source_sha256'], 'supplement_source_stability')
    audit.require(manifest['main_config_sha256'] == supplement['main_config_sha256'] == sha(ROOT/'config/ceiling_benchmark.json'), 'main_config_identity')
    declaration_path = ROOT/'evidence/ceiling_supplement_pre_outcome_freeze.json'
    declaration = read(declaration_path)
    audit.require(freeze['predeclaration_record_sha256'] == sha(declaration_path), 'predeclaration_record_identity')
    audit.require(declaration['supplement_config_sha256'] == sha(ROOT/'config/ceiling_supplement_checks.json'), 'supplement_predeclared_config_identity')
    audit.require(not declaration['main_check_scores_inspected_by_root'] and declaration['frozen_at_utc'] < freeze['started_at_utc'], 'supplement_prospective_scope')
    audit.require(manifest['counts'] == {'mesh_conditions': 48, 'calibration_information_conditions': 64, 'operational_conditions': 64}, 'frozen_supplement_counts')
    mesh, exact, operational, geometry = [read(OUT/(name+'.json')) for name in ['mesh_sensitivity', 'calibration_information', 'calibration_operational', 'geometry_checks']]
    audit.require([len(mesh), len(exact), len(operational)] == [48,64,64], 'actual_supplement_counts')
    banks = {}
    for name in config['geometry_banks']:
        bank = npz(ROOT/'resources/ceiling_banks_v1'/(name+'.npz'))
        bank['parameters'] = read(ROOT/'resources/ceiling_banks_v1'/(name+'_parameters.json'))
        banks[name] = bank
    metadata = read(ROOT/'results/ceiling_benchmark_v1/case_metadata.json')
    states = {(r['bank'], r['task'], r['domain']):r['states'] for r in metadata}
    source_checks, native_checks, fine_inputs = {}, {}, {}
    older_bem_manifest = read(ROOT/'resources/bem_validation/manifest.json')
    fsfine = ROOT/'resources/anatomy/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif'
    audit.require(sha(fsfine) == older_bem_manifest['input_sha256'][str(fsfine.relative_to(ROOT))], 'official_fsaverage_finer_solution_hash')
    fine_inputs[str(fsfine.relative_to(ROOT))] = sha(fsfine)
    for subject in ['fsaverage', 'sample']:
        bank_name = 'bem_'+subject; bank = banks[bank_name]
        position, normals, source_path, transform_path = official_source_identity(subject)
        audit.close('official_source_position_identity:'+subject, bank['source_positions_m'], position, atol=1e-9)
        audit.close('official_source_normal_identity:'+subject, bank['source_orientations'], normals, atol=1e-6)
        source_checks[bank_name] = {'sources':len(position), 'position_units':'m', 'coordinate_frame':'head',
            'independent_source_position_max_error_m':float(np.max(np.abs(bank['source_positions_m']-position))),
            'independent_source_normal_max_error':float(np.max(np.abs(bank['source_orientations']-normals))),
            'fine_forward_identity_check':'Frozen supplement code explicitly rejects changed source coordinates/normals; fine forward arrays not serialized'}
        for path in [source_path, transform_path]: fine_inputs[str(path.relative_to(ROOT))] = sha(path)
        gains = npz(OUT/(bank_name+'_mesh_nominal_gains.npz'))
        audit.close('coarse_gain_identity:'+subject, gains['ico3'], bank['nominal_gain_v_per_am'], atol=0., rtol=0.)
        audit.require(gains['ico3'].shape == gains['ico4'].shape == (64,32) and np.isfinite(gains['ico4']).all(), 'mesh_gain_shape_finite:'+subject)
        car = np.eye(64)-np.ones((64,64))/64
        relative = np.linalg.norm(car@(gains['ico4']-gains['ico3'])) / np.linalg.norm(car@gains['ico4'])
        audit.close('mesh_relative_car_gain:'+subject, geometry[bank_name]['ico3_to_ico4_relative_car_gain'], relative)
        audit.require(geometry[bank_name]['triangles_per_layer'] == [5120,5120,5120], 'fine_triangle_counts:'+subject)
        for row in [r for r in mesh if r['bank'] == bank_name]:
            task = row['task']; parameters = states[bank_name,task,'check_id']
            results = []
            for level in ['ico3','ico4']:
                current = dict(bank); current['gains_v_per_am'] = np.broadcast_to(gains[level], bank['gains_v_per_am'].shape)
                source = independent.reconstruct_source(current, task, parameters, config)
                law = independent.physical_law(current, source, 19, config['noise_regimes'][row['noise']], config)
                results.append(independent.oracle_law(law)['information_bits'])
            difference = results[1]-results[0]
            for field, expected in [('ico3_known_state_bits',results[0].mean()),('ico4_known_state_bits',results[1].mean()),
                                    ('difference_bits',difference.mean()),('maximum_absolute_source_state_difference_bits',np.max(np.abs(difference)))]:
                audit.close('mesh_information_recomputation:'+bank_name+task+row['noise']+field,row[field],expected,atol=2e-10)
            audit.require('not physiological accuracy' in row['scope'] and 'model/resolution sensitivity' in row['scope'], 'mesh_model_resolution_scope:'+bank_name)
            audit.counts['mesh_rows_recomputed'] += 1
    base = ROOT/'resources/ceiling_anatomy/sample'
    coarse_surfaces = mne.read_bem_surfaces(base/'bem/sample-1280-1280-1280-bem.fif',verbose=False)
    tree = read(Path('/tmp/tdo_ceiling_testing_tree.json'))
    git_blobs = {r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}
    binary_ascii, coarse_native = [], []
    for name, identity in [('outer_skin',4),('outer_skull',3),('inner_skull',1)]:
        binary_path, ascii_path = base/'bem_native'/(name+'.surf'), base/'bem_native'/(name+'.tri')
        binary, faces = binary_surface(binary_path)
        with ascii_path.open() as stream:
            count = int(stream.readline())
            ascii_coordinates = np.array([[float(v) for v in stream.readline().split()[1:]] for _ in range(count)])*.001
        audit.require(binary.shape == ascii_coordinates.shape == (2562,3) and faces.shape == (5120,3), 'native_surface_counts:'+name)
        audit.require(np.isfinite(binary).all() and faces.min() >= 0 and faces.max() < len(binary), 'native_surface_validity:'+name)
        discrepancy = float(np.max(np.abs(binary-ascii_coordinates))); binary_ascii.append(discrepancy)
        old = next(s for s in coarse_surfaces if s['id']==identity)
        distance = float(cKDTree(binary).query(old['rr'])[0].max()); coarse_native.append(distance)
        recorded = geometry['bem_sample']['native_surfaces'][name]
        audit.require(sha(binary_path)==recorded['sha256'], 'native_surface_recorded_sha:'+name)
        for path in [binary_path,ascii_path]:
            data = path.read_bytes(); gitsha = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            audit.require(gitsha==git_blobs['subjects/sample/bem/'+path.name], 'official_native_git_blob:'+path.name)
            fine_inputs[str(path.relative_to(ROOT))]=sha(path)
        native_checks[name]={'binary_ascii_coordinate_difference_m':discrepancy,'coarse_native_nearest_distance_m':distance,'vertices':len(binary),'triangles':len(faces)}
    audit.close('binary_ascii_coordinates',geometry['bem_sample']['max_binary_ascii_surface_coordinate_difference_m'],max(binary_ascii),atol=1e-15)
    audit.close('coarse_native_model_difference',geometry['bem_sample']['max_official_coarse_to_native_nearest_vertex_distance_m'],max(coarse_native),atol=1e-13)
    audit.require(max(binary_ascii)<1e-7 and 'not pure mesh convergence' in geometry['bem_sample']['sensitivity_scope'], 'native_units_and_model_scope')
    derived = base/'bem/sample-derived-5120-5120-5120-bem-sol.fif'
    derived_record = read(derived.with_suffix('.json'))
    audit.require(sha(derived)==geometry['bem_sample']['derived_solution_sha256']==derived_record['sha256'], 'derived_finer_solution_identity')
    fine_inputs[str(derived.relative_to(ROOT))]=sha(derived)
    exact_index = {(r['bank'],r['task'],r['domain'],r['calibration_epochs']):r for r in exact}
    expected_keys = {(bank,task,domain,n) for bank in banks for task in supplement['calibration']['tasks'] for domain in supplement['calibration']['domains'] for n in supplement['calibration']['known_label_epoch_budgets']}
    audit.require(set(exact_index)==expected_keys and len(exact_index)==64,'exact_calibration_grid')
    simultaneous_multiplicity = 16*len(exact)
    conditional_intervals = {}
    for key,row in exact_index.items():
        bank,task,domain,n = key; draws=supplement['calibration']['exact_information_outer_draws']
        audit.require(row['n_calibration']==n and row['full']['n_samples']==draws and row['paired_calibration_information_gain']['n_outer_draws']==draws, 'exact_calibration_counts:'+str(key))
        audit.require(row['seed']==independent.seed(config,supplement['seed_salt'],'calibration',bank,task,domain,n), 'exact_calibration_seed:'+str(key))
        for name in ['full','known_head_paired']:
            summary=row[name]
            check_information_summary(audit,str(key)+name,summary,1.,0.,1.,draws)
            half=.5*np.sqrt(np.log(40)/(2*draws))
            audit.close('exact_risk_lower:'+str(key)+name,summary['bayes_error_confidence_lower'],max(0.,summary['bayes_error']-half))
            audit.close('exact_risk_upper:'+str(key)+name,summary['bayes_error_confidence_upper'],min(.5,summary['bayes_error']+half))
        for field,width,low,high in [('paired_calibration_information_gain',2.,-1.,1.),('acquisition_information_loss',2.,-1.,1.),('acquisition_bayes_risk_increase',1.,-.5,.5)]:
            check_information_summary(audit,str(key)+field,row[field],width,low,high,draws)
            interval=independent.bounded_saved(row[field]['estimate'],row[field]['standard_error'],draws,low,high,simultaneous_multiplicity)
            for method,(lower,upper) in interval.items():
                if upper < -1e-12: audit.flags.append({'condition':list(key),'endpoint':field,'interval':method,'upper':upper})
        audit.close('exact_acquisition_MI_algebra:'+str(key),row['acquisition_information_loss']['estimate'],row['known_head_paired']['bits']-row['full']['bits'])
        audit.close('exact_acquisition_risk_algebra:'+str(key),row['acquisition_bayes_risk_increase']['estimate'],row['full']['bayes_error']-row['known_head_paired']['bayes_error'])
        law_path=ROOT/'results/ceiling_benchmark_v1/inputs'/f'{bank}__{task}__joint__{domain}.npz'
        law=npz(law_path); expected=independent.oracle_law(law)
        independent.oracle_check(audit,str(key),expected,row['known_head_exact'])
        ci=independent.bounded_saved(row['full']['bits'],row['full']['standard_error'],draws,0.,1.,simultaneous_multiplicity)
        risk_ci=independent.bounded_saved(row['full']['bayes_error'],row['full']['bayes_error_standard_error'],draws,0.,.5,simultaneous_multiplicity)
        conditional_intervals[key]={'information':ci,'risk':risk_ci}
        for method,(lower,upper) in ci.items():
            if lower>expected['information_bits'].mean()+1e-10:audit.flags.append({'condition':list(key),'endpoint':'conditional_information_exceeds_state_known','interval':method,'lower':lower})
        for method,(lower,upper) in risk_ci.items():
            if upper<expected['bayes_errors'].mean()-1e-10:audit.flags.append({'condition':list(key),'endpoint':'conditional_risk_below_state_known','interval':method,'upper':upper})
        if n==0:
            audit.close('zero_calibration_gain:'+str(key),[row['paired_calibration_information_gain']['estimate'],row['paired_calibration_information_gain']['standard_error']],[0.,0.],atol=1e-14)
            audit.close('zero_calibration_head_entropy:'+str(key),row['head_posterior_entropy_bits'],np.log2(24))
        audit.require(0<=row['head_posterior_entropy_bits']<=np.log2(24)+1e-10,'posterior_head_entropy_range:'+str(key))
        audit.require('same head across epochs' in row['estimand'] and 'finite-prior oracle knows latent prototypes' in row['scope'],'exact_calibration_scope:'+str(key))
        audit.counts['exact_summary_conditions_checked']+=1
    for bank in banks:
        for task in supplement['calibration']['tasks']:
            for domain in supplement['calibration']['domains']:
                budgets=supplement['calibration']['known_label_epoch_budgets']
                for small,large in zip(budgets,budgets[1:]):
                    left,right=conditional_intervals[bank,task,domain,small],conditional_intervals[bank,task,domain,large]
                    for method in left['information']:
                        if left['information'][method][0]>right['information'][method][1]+1e-12:
                            audit.flags.append({'condition':[bank,task,domain],'endpoint':'calibration_information_expected_monotonicity','budgets':[small,large],'interval':method})
                        if right['risk'][method][0]>left['risk'][method][1]+1e-12:
                            audit.flags.append({'condition':[bank,task,domain],'endpoint':'calibration_risk_expected_monotonicity','budgets':[small,large],'interval':method})
    operational_keys=set()
    for row in operational:
        key=(row['bank'],row['task'],row['domain'],row['calibration_epochs']);operational_keys.add(key)
        errors,losses,heads=[np.asarray(row[name]) for name in ['subject_errors','subject_log_losses_bits','subject_indices']]
        audit.require(errors.shape==losses.shape==heads.shape==(256,) and row['n_subjects']==256 and row['test_epochs_per_subject']==32,'operational_independent_subject_counts:'+str(key))
        audit.require(np.all((heads>=0)&(heads<24)) and np.issubdtype(heads.dtype,np.integer),'operational_state_indices:'+str(key))
        audit.require(np.all((errors>=0)&(errors<=1)) and np.all((losses>=0)&(losses<=-np.log2(row['probability_clip']))),'operational_clip_ranges:'+str(key))
        audit.close('operational_epoch_fraction:'+str(key),errors*32,np.rint(errors*32),atol=1e-13)
        for field,values,low,high in [('error',errors,0.,1.),('log_loss_bits',losses,0.,-np.log2(1e-6))]:
            independent.bounded_check(audit,str(key)+field,row[field],values,low,high)
            audit.require(row[field]['n_outer_draws']==256 and row[field]['multiplicity']==1,'operational_uncertainty_unit:'+str(key)+field)
        expected_random_seed=independent.seed(config,supplement['seed_salt'],'calibration',*key[:3],key[3])+3
        expected_heads=np.random.default_rng(expected_random_seed).choice(24,256,p=np.full(24,1/24))
        audit.require(np.array_equal(heads,expected_heads),'operational_subject_state_draw_replay:'+str(key))
        check_ids=set(npz(ROOT/'results/ceiling_benchmark_v1/inputs'/f'{key[0]}__{key[1]}__joint__{key[2]}.npz')['head_ids'].tolist())
        dev_ids=set(npz(ROOT/'results/ceiling_benchmark_v1/inputs'/f'{key[0]}__{key[1]}__joint__development.npz')['head_ids'].tolist())
        audit.require(not check_ids&dev_ids,'development_operational_state_disjointness:'+str(key))
        audit.require('not true MI' in row['scope'],'operational_misspecified_prior_scope:'+str(key))
        audit.counts['operational_conditions_recomputed']+=1
    audit.require(operational_keys==expected_keys and len(operational)==len(operational_keys),'operational_frozen_grid')
    output={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started,
        'passed':not audit.failures,'failures':audit.failures,'numerical_flags':audit.flags,'hash_checks':audit.hashes,
        'maximum_absolute_residuals':audit.maximum_residual,'recomputed_counts':dict(audit.counts),
        'source_identity_checks':source_checks,'native_geometry_checks':native_checks,'finer_input_sha256':fine_inputs,
        'numerical_inequality_scope':{'confidence':.95,'multiplicity':simultaneous_multiplicity,
            'methods':'Separate global Bonferroni Hoeffding and iid empirical-Bernstein checks; no strict point-estimate monotonicity'},
        'sampling_contract':'Exact information averages independent outer calibration/test subjects. Operational sampling independently draws256 subjects, each with one shared latent state/calibration and32 test epochs; uncertainty uses256 subject means, not8192 repeated epochs.',
        'limitations':['Exact-calibration JSON serializes summaries, not per-draw integrands; paired-gain SE cannot be fully recomputed here. A separately hashed same-seed replay is planned.',
            'Fine-forward source coordinates/normals were checked by explicit frozen runtime assertions; official source reconstruction independently agrees with main arrays, but fine-output coordinate arrays are not serialized.',
            'Sample coarse/finer models differ in supplied geometry/smoothing as well as discretization; comparison is model-plus-resolution sensitivity, not pure convergence or physiological accuracy.',
            'Operational subject-unit arrays allow complete risk/logloss/uncertainty recomputation, but per-epoch predictions and calibration records are not serialized in this supplement.',
            'Repeated simulated states are independent subject draws from a finite bank, not distinct human anatomies or a continuous-population guarantee.'],
        'audit_source_sha256':sha(__file__),'independent_calculation_source_sha256':sha(ROOT/'scripts/verify_ceiling_benchmark_outputs.py'),
        'supplement_manifest_sha256':sha(OUT/'manifest.json'),'main_manifest_sha256':sha(ROOT/'results/ceiling_benchmark_v1/run_manifest.json'),
        'packages':{'numpy':np.__version__,'mne':mne.__version__}}
    EVIDENCE.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':output['passed'],'failures':len(audit.failures),'numerical_flags':len(audit.flags),'seconds':output['elapsed_seconds'],'evidence':str(EVIDENCE)},indent=2))
    if audit.failures:sys.exit(1)


if __name__=='__main__':main()
