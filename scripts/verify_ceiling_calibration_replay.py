#!/usr/bin/env python3
"""Independent replay-array and presentation audit; all study files read-only."""
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
OUT=ROOT/'results/ceiling_benchmark_v1_calibration_replay'
SUP=ROOT/'results/ceiling_benchmark_v1_supplements'
PRESENT=ROOT/'results/ceiling_benchmark_v1_presentation'
EVIDENCE=ROOT/'evidence/ceiling_calibration_replay_integrity_checks.json'
spec=importlib.util.spec_from_file_location('independent_main_audit',ROOT/'scripts/verify_ceiling_benchmark_outputs.py')
independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)


def exact_summary(audit,name,saved,information,risk):
    independent.summary_check(audit,name,saved,information,risk)
    # The original supplement exposes pointwise Hoeffding only, so fields
    # absent there are not retrospectively asserted as original outputs.
    audit.require(saved['n_samples']==len(information),'exact_draw_count:'+name)


def aggregate_pairs(audit,decisions,statistics,domain):
    families=sorted({r['case_key'] for r in decisions if r['domain']==domain and r['budget']==64})
    differences=[]
    for key in families:
        choices={r['policy']:r['candidate'] for r in decisions if r['case_key']==key and r['budget']==64}
        data=independent.npz(ROOT/'results/ceiling_benchmark_v1/integration'/(key+'.npz'))
        difference=data[choices['fixed_generic_pca']+'__information']-data[choices['development_information']+'__information']
        audit.require(difference.shape==(16384,),'aggregate_stratum_size:'+key)
        differences.append(difference)
    array=np.concatenate(differences)
    # Independent nonidentical strata with fixed equal sample counts. This
    # Hoeffding bound requires independence, not an iid mixture assumption.
    half=2*np.sqrt(np.log(4/.05)/(2*len(array)))
    saved=next(r for r in statistics if r['domain']==domain)
    expected={'mean':float(array.mean()),'interval':[float(array.mean()-half),float(array.mean()+half)],
        'n_cases':len(families),'n_draws':len(array),'range_width':2.,'two_domain_bonferroni':2}
    audit.close('paired_grid_mean:'+domain+str(len(families)),saved['fixed_generic_minus_development_bits'],expected['mean'])
    audit.close('paired_grid_interval:'+domain+str(len(families)),saved['joint_numerical_ci95'],expected['interval'])
    audit.require(saved['n_cases']==len(families) and saved['n_independent_integration_draws']==len(array),'paired_grid_counts:'+domain+str(len(families)))
    return expected


def main():
    start=time.perf_counter();audit=independent.Audit()
    read,sha,npz=independent.read,independent.sha,independent.npz
    manifest=read(OUT/'manifest.json');supp_manifest=read(SUP/'manifest.json')
    config=read(ROOT/'config/ceiling_benchmark.json')
    audit.mapping('replay_inputs_and_sources',manifest['input_source_sha256'])
    audit.mapping('replay_outputs',manifest['output_sha256'])
    audit.mapping('unchanged_supplement_outputs',supp_manifest['output_sha256'])
    audit.mapping('unchanged_supplement_sources',supp_manifest['source_sha256'])
    main_manifest=read(ROOT/'results/ceiling_benchmark_v1/run_manifest.json')
    audit.mapping('unchanged_main_outputs',main_manifest['output_sha256'])
    audit.mapping('unchanged_main_sources',main_manifest['source_sha256_after'])
    rows=read(SUP/'calibration_information.json');matches=read(OUT/'matches.json')
    audit.require(len(rows)==len(matches)==manifest['conditions']==64,'frozen_replay_count')
    by_key={(r['bank'],r['task'],r['domain'],r['calibration_epochs']):r for r in rows}
    match_by_key={(r['bank'],r['task'],r['domain'],r['calibration_epochs']):r for r in matches}
    audit.require(set(by_key)==set(match_by_key) and len(by_key)==64,'replay_condition_identity')
    interval_family=1024;conditional_intervals={};maximum_replay_residual=0.;seeds=[]
    for key,row in by_key.items():
        match=match_by_key[key];a=npz(ROOT/match['path']);n=len(a['heads']);budget=key[3]
        audit.require(n==16384 and a['labels'].shape==(n,) and a['calibration_labels'].shape==(n,budget),'persistent_state_draw_shapes:'+str(key))
        audit.require(np.all(np.isin(a['labels'],[-1,1])) and np.all(np.isin(a['calibration_labels'],[-1,1])),'known_label_values:'+str(key))
        audit.require(np.all((a['heads']>=0)&(a['heads']<24)),'finite_bank_state_indices:'+str(key))
        audit.require(match['seed']==row['seed'],'same_seed_replay:'+str(key));seeds.append(row['seed'])
        rng=np.random.default_rng(row['seed'])
        expected_heads=rng.choice(24,n,p=np.full(24,1/24));expected_calibration_labels=rng.integers(0,2,(n,budget))*2-1
        audit.require(np.array_equal(a['heads'],expected_heads) and np.array_equal(a['calibration_labels'],expected_calibration_labels),'head_and_calibration_rng_replay:'+str(key))
        exact_summary(audit,str(key)+'full',row['full'],a['full_information'],a['full_risk'])
        exact_summary(audit,str(key)+'known',row['known_head_paired'],a['known_information'],a['known_risk'])
        audit.close('zero_posterior_entropy_risk_identity:'+str(key),a['zero_information'],1-independent.entropy(a['zero_risk']),atol=2e-13)
        gain=a['full_information']-a['zero_information']
        acquisition=a['known_information']-a['full_information']
        risk_increase=a['full_risk']-a['known_risk']
        for field,values,low,high in [('paired_calibration_information_gain',gain,-1.,1.),('acquisition_information_loss',acquisition,-1.,1.),
                                     ('acquisition_bayes_risk_increase',risk_increase,-.5,.5)]:
            independent.bounded_check(audit,str(key)+field,row[field],values,low,high)
            summary=independent.bounded(values,low,high,interval_family)
            for method,upper in [('Hoeffding',summary['confidence_upper']),('empirical_Bernstein',summary['empirical_bernstein_upper'])]:
                if upper < -1e-12:audit.flags.append({'condition':list(key),'endpoint':field,'interval':method,'upper':upper})
        audit.close('state_posterior_entropy:'+str(key),row['head_posterior_entropy_bits'],a['state_posterior_entropy'].mean())
        audit.require(a['state_posterior_entropy'].shape==(n,) and np.all((a['state_posterior_entropy']>=0)&(a['state_posterior_entropy']<=np.log2(24)+1e-12)),'posterior_state_entropy_range:'+str(key))
        law=npz(ROOT/'results/ceiling_benchmark_v1/inputs'/f'{key[0]}__{key[1]}__joint__{key[2]}.npz')
        expected_oracle=independent.oracle_law(law)
        independent.oracle_check(audit,str(key),expected_oracle,row['known_head_exact'])
        conditional_intervals[key]={}
        for endpoint,values,low,high,known in [('information',a['full_information'],0.,1.,expected_oracle['information_bits'].mean()),
                                             ('risk',a['full_risk'],0.,.5,expected_oracle['bayes_errors'].mean())]:
            summary=independent.bounded(values,low,high,interval_family);conditional_intervals[key][endpoint]=summary
            for method,lower,upper in [('Hoeffding',summary['confidence_lower'],summary['confidence_upper']),
                                      ('empirical_Bernstein',summary['empirical_bernstein_lower'],summary['empirical_bernstein_upper'])]:
                violation=lower>known+1e-10 if endpoint=='information' else upper<known-1e-10
                if violation:audit.flags.append({'condition':list(key),'endpoint':'conditional_'+endpoint+'_known_state_bound','interval':method,'known':float(known)})
        if budget==0:
            audit.close('cal0_information_identity:'+str(key),a['full_information'],a['zero_information'],atol=1e-14)
            audit.close('cal0_risk_identity:'+str(key),a['full_risk'],a['zero_risk'],atol=1e-14)
            audit.close('cal0_state_entropy_identity:'+str(key),a['state_posterior_entropy'],np.full(n,np.log2(24)),atol=1e-13)
        residuals={
            'full_bits':abs(a['full_information'].mean()-row['full']['bits']),
            'full_risk':abs(a['full_risk'].mean()-row['full']['bayes_error']),
            'known_bits':abs(a['known_information'].mean()-row['known_head_paired']['bits']),
            'known_risk':abs(a['known_risk'].mean()-row['known_head_paired']['bayes_error']),
            'paired_gain':abs(gain.mean()-row['paired_calibration_information_gain']['estimate']),
            'paired_gain_se':abs(gain.std(ddof=1)/np.sqrt(n)-row['paired_calibration_information_gain']['standard_error']),
            'posterior_entropy':abs(a['state_posterior_entropy'].mean()-row['head_posterior_entropy_bits'])}
        for field,value in residuals.items():audit.close('replay_match_residual:'+str(key)+field,match['residuals'][field],value,atol=5e-15)
        maximum_replay_residual=max(maximum_replay_residual,max(residuals.values()))
        audit.counts['calibration_replay_conditions']+=1
        audit.counts['independent_outer_calibration_draws']+=n
    audit.require(len(set(seeds))==64,'all_replay_case_seeds_unique')
    audit.close('declared_maximum_replay_residual',manifest['maximum_summary_residual'],maximum_replay_residual,atol=5e-15)
    for bank in config['geometry_banks']:
        for task in config['decoder_tasks']:
            for domain in ['check_id','check_shift']:
                for small,large in zip([0,2,8],[2,8,32]):
                    si,li=conditional_intervals[bank,task,domain,small],conditional_intervals[bank,task,domain,large]
                    for lower,upper,method in [('confidence_lower','confidence_upper','Hoeffding'),('empirical_bernstein_lower','empirical_bernstein_upper','empirical_Bernstein')]:
                        if si['information'][lower]>li['information'][upper]+1e-12:
                            audit.flags.append({'family':[bank,task,domain],'budgets':[small,large],'endpoint':'expected_calibration_MI_monotonicity','interval':method})
                        if li['risk'][lower]>si['risk'][upper]+1e-12:
                            audit.flags.append({'family':[bank,task,domain],'budgets':[small,large],'endpoint':'expected_calibration_risk_monotonicity','interval':method})
    presentation_manifest=read(PRESENT/'manifest.json');statistics=read(PRESENT/'statistics.json')
    audit.mapping('presentation_inputs',presentation_manifest['input_sha256'])
    audit.mapping('presentation_outputs',presentation_manifest['output_sha256'])
    audit.require(presentation_manifest['script_sha256']==sha(ROOT/'scripts/report_ceiling_benchmark.py'),'presentation_source_identity')
    transfer=read(ROOT/'results/ceiling_benchmark_v1/anatomy_transfer_conditions.json')
    tdec=read(ROOT/'results/ceiling_benchmark_v1/anatomy_transfer_decisions.json')
    local=read(ROOT/'results/ceiling_benchmark_v1/decisions.json')
    aggregates={'transfer':{},'within_family':{}}
    for domain in ['check_id','check_shift']:
        aggregates['transfer'][domain]=aggregate_pairs(audit,tdec,statistics['transfer_paired_grid_comparisons'],domain)
        aggregates['within_family'][domain]=aggregate_pairs(audit,local,statistics['within_family_paired_grid_comparisons'],domain)
    for shown in statistics['anatomy_transfer']:
        key=(shown['task'],shown['domain']);candidate=shown['candidate']
        row=next(r for r in transfer if (r['task'],r['domain'],r['candidate'])==(*key,candidate))
        encoded=row['representation_information']['known_head_exact'];full=row['known_state_reference']
        for field,value in [('sensor_bits',row['sensor_bits']),('selected_bits',row['representation_bits']),
                            ('selected_state_known_bits',encoded['average_information_bits']),('selected_bayes_error',row['representation_information']['bayes_error'])]:
            audit.close('report_oracle_label_and_value:'+str(key)+field,shown[field],value)
        expected_ratio=np.median(np.array(encoded['separations'])**2/np.maximum(np.array(full['separations'])**2,1e-300))
        audit.close('report_encoded_full_separation_ratio:'+str(key),shown['state_known_squared_separation_retention_median'],expected_ratio)
        audit.require(encoded['average_information_bits']<=full['average_information_bits']+1e-10,'encoded_oracle_vs_full_oracle:'+str(key))
    report=(ROOT/'reports/ceiling_benchmark_results.md').read_text()
    audit.require('fixed encoder' in report and 'forward/source-family transfer' in report and 'not an isolated causal intervention on head shape alone' in report,
        'report_calibration_and_transfer_scope')
    audit.require('pointwise95% Hoeffding' in report and 'not familywise intervals' in report and 'not a convergence proof or physical validation' in report,
        'report_numerical_and_geometry_scope')
    # Hardcoded prose values are independently checked too, separately from
    # the generated tables. Rounding tolerances match displayed decimal places.
    for domain,generic_mean,selected_mean in [('check_id',.06872,.01168),('check_shift',.05297,.00770)]:
        shown=[r for r in statistics['anatomy_transfer'] if r['domain']==domain]
        audit.close('prose_mean_generic:'+domain,generic_mean,np.mean([r['fixed_generic_bits'] for r in shown]),atol=5e-6,rtol=0.)
        audit.close('prose_mean_template:'+domain,selected_mean,np.mean([r['selected_bits'] for r in shown]),atol=5e-6,rtol=0.)
    lateral=next(r for r in transfer if r['task']=='spatial_lateral' and r['domain']=='check_id' and r['candidate']==
        next(r['candidate'] for r in tdec if r['task']=='spatial_lateral' and r['domain']=='check_id' and r['policy']=='development_information'))
    for field,value,precision in [('sensor_bits',.17593,5e-6),('representation_bits',.00665,5e-6)]:
        audit.close('prose_lateral_'+field,lateral[field],value,atol=precision,rtol=0.)
    audit.close('prose_lateral_encoded_oracle',lateral['representation_information']['known_head_exact']['average_information_bits'],.00720,atol=5e-6,rtol=0.)
    audit.close('prose_lateral_encoded_error',100*lateral['representation_information']['bayes_error'],46.17,atol=.005,rtol=0.)
    audit.close('prose_lateral_full_error',100*lateral['sensor_information']['bayes_error'],29.92,atol=.005,rtol=0.)
    resolution_audit_path=ROOT/'evidence/ceiling_transfer_resolution_integrity_checks.json'
    resolution_audit=read(resolution_audit_path)
    audit.require(resolution_audit['passed'] and not resolution_audit['failures'] and not resolution_audit['numerical_flags'],
        'separate_transfer_resolution_audit_passed')
    resolution_rows=statistics['post_outcome_transfer_resolution_sensitivity']
    audited_resolution={(r['task'],r['domain'],r['mesh']):r for r in resolution_audit['independently_recomputed_results']}
    audit.require(len(resolution_rows)==len(audited_resolution)==24,'presentation_resolution_grid')
    for shown in resolution_rows:
        key=(shown['task'],shown['domain'],shown['mesh']);computed=audited_resolution[key]
        for field,target in [('full_state_known_bits','full_state_known_bits'),
                             ('template_choice_state_known_bits','development_information_bits'),
                             ('fixed_generic_state_known_bits','fixed_generic_pca_bits')]:
            audit.close('presentation_resolution_oracle:'+str(key)+field,shown[field],computed[target])
    for mesh,encoded,full in [('ico3',.00743,.18427),('ico4',.00749,.16813)]:
        computed=audited_resolution['spatial_lateral','check_id',mesh]
        audit.close('resolution_lateral_prose_encoded:'+mesh,computed['development_information_bits'],encoded,atol=5e-6,rtol=0.)
        audit.close('resolution_lateral_prose_full:'+mesh,computed['full_state_known_bits'],full,atol=5e-6,rtol=0.)
    audit.require(all(audited_resolution[task,'check_id','ico4']['development_information_bits']<.0012
        for task in config['tasks'] if task.startswith('temporal_')),'resolution_temporal_prose_ceiling')
    audit.require('post-outcome diagnostic' in report and 'restricted to nominal caps' in report
        and 'persistent source-amplitude factor' in report,'report_final_sensitivity_and_amplitude_scope')
    output={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,
        'passed':not audit.failures,'failures':audit.failures,'numerical_flags':audit.flags,'hash_checks':audit.hashes,
        'maximum_absolute_residuals':audit.maximum_residual,'recomputed_counts':dict(audit.counts),
        'maximum_summary_match_residual':float(maximum_replay_residual),'paired_grid_aggregates':aggregates,
        'report_review':{'encoded_state_known_ceiling':'representation_information.known_head_exact, correctly used in transfer table/diamonds',
            'full_state_known_ceiling':'known_state_reference, correctly used for full observation and separation-retention denominator',
            'aggregate_uncertainty':'Same-case paired candidate differences; independent fixed-size finite-case strata. Two domains jointly bounded using range2 and Bonferroni2; no anatomy-population interval.',
            'calibration_gains':'Per-row paired calibrated-minus-unconditional information on identical test draws; separately integrated cal0 means are not substituted.',
            'post_outcome_resolution':'24 presentation rows agree with independently reconstructed coarse/fine raw laws and frozen encoder oracles; prose limits this to nominal caps and model-plus-resolution sensitivity.'},
        'inequality_scope':{'confidence':.95,'multiplicity':interval_family,'methods':'Hoeffding and empirical Bernstein checked separately; finite-MC point monotonicity never imposed'},
        'limitations':['Same-seed replay uses the frozen channel implementation; this audit independently recomputes every stored statistic, but does not independently regenerate all conditional likelihoods.',
            'General likelihood correctness is separately supported by dense-Gaussian unit/math audits and actual main-case1152D/64D independent replays.',
            'The replay closes summary/integrand serialization, not uncertainty about continuous acquisition/source laws or anatomy populations.',
            'Presentation averages have equal frozen-case weights and numerical uncertainty; successful local forecasts do not establish globally optimal or anatomically universal encoders.'],
        'audit_source_sha256':sha(__file__),'independent_calculation_source_sha256':sha(ROOT/'scripts/verify_ceiling_benchmark_outputs.py'),
        'replay_manifest_sha256':sha(OUT/'manifest.json'),'presentation_manifest_sha256':sha(PRESENT/'manifest.json'),
        'separate_transfer_resolution_audit_sha256':sha(resolution_audit_path),
        'interpreter':sys.executable,'packages':{'numpy':np.__version__}}
    EVIDENCE.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':output['passed'],'failures':len(audit.failures),'flags':len(audit.flags),'seconds':output['elapsed_seconds'],'evidence':str(EVIDENCE)},indent=2))
    if audit.failures:sys.exit(1)


if __name__=='__main__':main()
