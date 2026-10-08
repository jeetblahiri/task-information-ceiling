#!/usr/bin/env python3
"""Frozen prospective information-ceiling/representation-retention benchmark."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr
from scipy.stats import spearmanr
from tdo_sim.coordinates import car_basis, covariance_whitener
from tdo_sim.noise import sensor_covariance, temporal_covariance
from tdo_sim.decoders import train_linear, train_mlp
from tdo_sim.ceiling_benchmark import (prepared_experiment,prepare_transform_experiment,
    block_average_matrix,paired_task_information,draw_raw_epochs,summarize_bounded)

CONFIG=ROOT/'config/ceiling_benchmark.json'
OUT=ROOT/'results/ceiling_benchmark_v1'
ASSETS=ROOT/'resources/ceiling_banks_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def dump(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')

def seed(config,*parts):
    # Reproducible identifier salt; draws remain independent across explicit stages.
    return int(np.random.SeedSequence([config['seed']]+[zlib.crc32(str(p).encode()) for p in parts]).generate_state(1)[0])

def load_banks(config):
    manifest=json.loads((ASSETS/'manifest.json').read_text())
    if manifest['config_sha256']!=sha(CONFIG):raise RuntimeError('Asset config mismatch')
    for name,digest in manifest['output_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Physical asset hash mismatch '+name)
    banks={}
    for i,name in enumerate(config['geometry_banks']):
        with np.load(ASSETS/(name+'.npz')) as f:bank={k:f[k] for k in f.files}
        bank.update(name=name,index=i,parameters=json.loads((ASSETS/(name+'_parameters.json')).read_text()),
                    metadata=json.loads((ASSETS/(name+'_metadata.json')).read_text()))
        banks[name]=bank
    return banks

def patch(positions,centre,width):
    # Peak amplitude is the source-coefficient vector L2 norm at waveform peak.
    squared=np.sum((positions-np.asarray(centre))**2,axis=1)
    vector=np.exp(-(squared-squared.min())/(2*width**2))
    return vector/np.linalg.norm(vector)

def source_means(bank,task_name,domain,config,amplitude=None):
    task=config['tasks'][task_name];state=config['source_states'][domain]
    split='check' if domain.startswith('check') else domain
    indices=np.array([i for i,p in enumerate(bank['parameters']) if p['split']==split])
    gains=bank['gains_v_per_am'][indices];positions=bank['source_positions_m']
    t=np.arange(config['n_times'])/config['sampling_frequency_hz']
    background_patch=patch(positions,config['background_centre_m'],config['background_width_m'])
    epoch_wave=np.cos(2*np.pi*8*(t-.12))*np.exp(-((t-.12)/.08)**2/2)
    epoch_wave/=np.max(np.abs(epoch_wave))
    static_wave=np.exp(-((t-.12)/.06)**2/2);static_wave/=static_wave.max()
    plus=[];minus=[];background=[];parameters=[]
    for local,index in enumerate(indices):
        rng=np.random.default_rng(seed(config,'source',bank['name'],task_name,domain,int(index)))
        jitter=rng.uniform(-state['centre_jitter_halfwidth_m'],state['centre_jitter_halfwidth_m'],3)
        factor=1+rng.uniform(-state['amplitude_halfwidth_fraction'],state['amplitude_halfwidth_fraction'])
        latency=rng.uniform(-state['latency_jitter_halfwidth_s'],state['latency_jitter_halfwidth_s'])
        width_factor=1+rng.uniform(-state['wave_width_halfwidth_fraction'],state['wave_width_halfwidth_fraction'])
        static_coefficient=float(rng.normal())
        background_spatial=gains[local]@background_patch
        common=config['static_background_peak_am']*static_coefficient*background_spatial[:,None]*static_wave
        means=[]
        for label in range(2):
            centre=np.asarray(task['centres_m'][label])+jitter
            weights=patch(positions,centre,task['patch_width_m'])
            mu=task['wave_centres_s'][label]+latency;width=task['wave_width_s']*width_factor
            wave=np.exp(-((t-mu)/width)**2/2)
            if 'frequencies_hz' in task:wave*=.5*(1+np.cos(2*np.pi*task['frequencies_hz'][label]*(t-mu)))
            wave/=wave.max()
            means.append((config['signal_peak_am'] if amplitude is None else amplitude)*factor*(gains[local]@weights)[:,None]*wave+common)
        minus.append(means[0]);plus.append(means[1])
        background.append(config['epoch_background_peak_am']*background_spatial[:,None]*epoch_wave)
        parameters.append({'head_id':bank['parameters'][index]['head_id'],'head_index':int(index),'state_seed':seed(config,'source',bank['name'],task_name,domain,int(index)),
            'common_source_jitter_m':jitter.tolist(),'source_amplitude_factor':factor,'common_latency_jitter_s':latency,
            'wave_width_factor':width_factor,'static_background_coefficient':static_coefficient,'fixed_across_observations':True})
    return {'plus':np.stack(plus),'minus':np.stack(minus),'background':np.stack(background),
            'head_ids':[p['head_id'] for p in parameters],'parameters':parameters}

def experiment(bank,state,regime,channels,config):
    indices=bank['montage_order'][:channels];q=car_basis(channels).T
    raw=sensor_covariance(bank['sensor_positions_m'],config['sensor_noise_sd_v'],regime['spatial_correlation'])
    cs=q@raw[np.ix_(indices,indices)]@q.T;ct=temporal_covariance(config['n_times'],regime['temporal_ar1'])
    observed=lambda x:np.einsum('ac,hct->hat',q,x[:,indices],optimize=True)
    return prepared_experiment(observed(state['plus']),observed(state['minus']),cs,ct,
        background=observed(state['background']),background_sd=1.,head_ids=state['head_ids'])

def moment(exp):
    means=np.concatenate((exp.plus,exp.minus));centre=means.mean(axis=0);residual=means-centre
    c=np.einsum('hct,hdt->cd',residual,residual,optimize=True)/(len(means)*means.shape[2])
    c+=exp.spatial_covariance*np.trace(exp.temporal_covariance)/means.shape[2]
    if exp.background is not None:
        u=exp.background*np.asarray(exp.background_sd)[:,None,None]
        c+=np.einsum('hct,hdt->cd',u,u,optimize=True)/(len(u)*means.shape[2])
    return (c+c.T)/2

def spatial_designs(dev,all_development):
    generic=sum(moment(e) for e in all_development)/len(all_development)
    # Ordinary multitask PCA centres the complete task mixture once. Add the
    # between-task mean covariance omitted by separately centred moments.
    centres=[np.concatenate((e.plus,e.minus)).mean(axis=0) for e in all_development]
    overall=sum(centres)/len(centres)
    generic+=sum((c-overall)@(c-overall).T/c.shape[1] for c in centres)/len(centres)
    matched=moment(dev)
    def eigenrows(c):
        _,v=np.linalg.eigh(c);return v[:,::-1].T
    effective=dev.spatial_covariance.copy()
    wt=covariance_whitener(dev.temporal_covariance)
    if dev.background is not None:
        u=dev.background*np.asarray(dev.background_sd)[:,None,None]
        temporal_white=np.einsum('hct,bt->hcb',u,wt,optimize=True)
        effective+=np.einsum('hct,hdt->cd',temporal_white,temporal_white,optimize=True)/(len(u)*u.shape[2])
    # Development-only separable spatial surrogate, not an exact optimizer
    # of the full per-state rank-one experiment. Actual MI uses exact laws.
    ws=covariance_whitener(effective)
    delta=dev.plus-dev.minus;white=np.einsum('ac,hct,bt->hab',ws,delta,wt,optimize=True)
    gram=np.einsum('hct,hdt->cd',white,white,optimize=True)/len(white)
    fisher=eigenrows(gram)@ws
    # Row rescaling changes coordinates only and avoids physical unit imbalance.
    fisher/=np.linalg.norm(fisher,axis=1)[:,None]
    return {'generic_pca':eigenrows(generic),'task_pca':eigenrows(matched),'task_fisher':fisher}

def variance_retention(exp,rs,tb):
    # Evaluate physical rowspaces, invariant to arbitrary feature row scaling.
    qs=rs.T@np.linalg.solve(rs@rs.T,rs);qt=tb.T@np.linalg.solve(tb@tb.T,tb)
    means=np.concatenate((exp.plus,exp.minus));means-=means.mean(axis=0)
    def energy(a,b):
        value=float(np.einsum('hct,cd,hdu,ut->',means,a,means,b,optimize=True)/len(means))
        value+=float(np.trace(a@exp.spatial_covariance)*np.trace(b@exp.temporal_covariance))
        if exp.background is not None:
            u=exp.background*np.asarray(exp.background_sd)[:,None,None]
            value+=float(np.einsum('hct,cd,hdu,ut->',u,a,u,b,optimize=True)/len(u))
        return value
    return energy(qs,qt)/energy(np.eye(qs.shape[0]),np.eye(qt.shape[0]))

def mean_prototype_oracle(exp):
    b=None if exp.background is None else exp.background.mean(axis=0,keepdims=True)
    proto=prepared_experiment(exp.plus.mean(axis=0,keepdims=True),exp.minus.mean(axis=0,keepdims=True),
        exp.spatial_covariance,exp.temporal_covariance,background=b,background_sd=1.)
    return proto.oracle_summary()

def info_bits(summary):
    for key in ['bits','average_information_bits','information_bits']:
        if key in summary and np.ndim(summary[key])==0:return float(summary[key])
    raise KeyError('Information summary has no scalar bits: '+str(summary.keys()))

def oracle_risk(summary):
    for key in ['average_bayes_error','bayes_error']:
        if key in summary:return float(summary[key])
    raise KeyError('Oracle risk absent')

def candidate_name(method,s,t,budget):return f'{method}_b{budget}_s{s}_t{t}'

def candidates(dev,all_development,config):
    bases=spatial_designs(dev,all_development);transforms={};specs={}
    for budget,pairs in config['feature_budgets'].items():
        for method in config['spatial_designs']:
            for s,t in pairs:
                name=candidate_name(method,s,t,budget)
                transforms[name]=(bases[method][:s],block_average_matrix(config['n_times'],config['n_times']//t))
                specs[name]={'method':method,'budget':int(budget),'spatial_features':s,'time_bins':t,
                             'bin_width_ms':1000/config['sampling_frequency_hz']*config['n_times']/t}
    return transforms,specs

def save_experiment(path,exp):
    np.savez_compressed(path,plus=exp.plus,minus=exp.minus,background=exp.background,background_sd=exp.background_sd,
        spatial_covariance=exp.spatial_covariance,temporal_covariance=exp.temporal_covariance,head_ids=np.array(exp.head_ids))

def save_arrays(path,result):
    arrays=result['sample_arrays'];out={'labels':arrays['labels'],'heads':arrays['head_indices']}
    for family in ['full','known_head']:
        for endpoint,value in arrays[family].items():out[family+'__'+endpoint]=value
    for name,row in arrays['representations'].items():
        for endpoint,value in row.items():out[name+'__'+endpoint]=value
    np.savez_compressed(path,**out)

def best(names,score):
    # Documented deterministic insertion-order tie rule, no check-driven tie tuning.
    maximum=max(score[n] for n in names)
    return next(n for n in names if maximum-score[n]<=1e-12)

def decide(transforms,specs,predicted,prototype,variance,config):
    selections=[]
    for budget in config['feature_budgets']:
        names=[n for n in transforms if specs[n]['budget']==int(budget)]
        if int(budget)==64:fixed=candidate_name('generic_pca',4,16,budget)
        elif int(budget)==32:fixed=candidate_name('generic_pca',4,8,budget)
        else:fixed=candidate_name('generic_pca',4,32,budget)
        policies={'development_information':best(names,predicted),'mean_prototype_information':best(names,prototype),
                  'variance_retention':best(names,variance),'fixed_generic_pca':fixed}
        for method in config['spatial_designs']:
            subset=[n for n in names if specs[n]['method']==method]
            policies['development_'+method]=best(subset,predicted)
        for policy,name in policies.items():selections.append({'budget':int(budget),'policy':policy,'candidate':name})
    return selections

def stage_designs(banks,config):
    designs={};predictions=[];state_records=[]
    for bank_name,bank in banks.items():
        states={task:source_means(bank,task,'development',config) for task in config['tasks']}
        for task,state in states.items():state_records.append({'bank':bank_name,'task':task,'domain':'development','states':state['parameters']})
        for regime_name in config['representation_noise_regimes']:
            regime=config['noise_regimes'][regime_name]
            devs={task:experiment(bank,state,regime,19,config) for task,state in states.items()}
            for task,dev in devs.items():
                key=f'{bank_name}__{task}__{regime_name}'
                trans,specs=candidates(dev,list(devs.values()),config)
                result=paired_task_information(dev,trans,n_samples=config['development_information_draws'],
                    seed=seed(config,'development_mi',key),batch_size=config['information_batch_size'],return_sample_arrays=True)
                predicted={n:info_bits(result['representations'][n]) for n in trans}
                prototype={};variance={}
                for name,(rs,tb) in trans.items():
                    compressed=prepare_transform_experiment(dev,rs,tb,name=name)
                    prototype[name]=info_bits(mean_prototype_oracle(compressed))
                    variance[name]=variance_retention(dev,rs,tb)
                    predictions.append({'design_key':key,'bank':bank_name,'task':task,'noise':regime_name,'candidate':name,**specs[name],
                        'development_full_bits':info_bits(result['full']),'predicted_bits':predicted[name],
                        'predicted_loss_bits':info_bits(result['full'])-predicted[name],
                        'mean_prototype_bits':prototype[name],'variance_retention_fraction':variance[name],
                        'information':result['representations'][name],'paired_loss':result['losses'][name]})
                selection=decide(trans,specs,predicted,prototype,variance,config)
                arrays={}
                for name,(rs,tb) in trans.items():arrays[name+'__spatial']=rs;arrays[name+'__temporal']=tb
                np.savez_compressed(OUT/'designs'/f'{key}.npz',**arrays)
                save_experiment(OUT/'inputs'/f'{key}__development.npz',dev)
                save_arrays(OUT/'integration'/f'{key}__development.npz',result)
                designs[key]={'transforms':trans,'specs':specs,'predicted':predicted,'predicted_full_bits':info_bits(result['full']),
                    'selections':selection,'development_experiment':dev}
                print('Development design saved '+key,flush=True)
    dump(OUT/'development_predictions.json',predictions)
    dump(OUT/'prospective_selections.json',{key:value['selections'] for key,value in designs.items()})
    dump(OUT/'development_state_records.json',state_records)
    dump(OUT/'development_stage_manifest.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'before_check_outcomes':True,'config_sha256':sha(CONFIG),'designs':len(designs),
        'design_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (OUT/'designs').glob('*.npz')},
        'predictions_sha256':sha(OUT/'development_predictions.json'),'selections_sha256':sha(OUT/'prospective_selections.json')})
    return designs,predictions

def check_family(bank,task,domain,regime_name,design,config,key_override=None,precision=False):
    state=source_means(bank,task,domain,config);regime=config['noise_regimes'][regime_name]
    exp=experiment(bank,state,regime,19,config)
    key=key_override or f'{bank["name"]}__{task}__{regime_name}__{domain}'
    trans=design['transforms']
    if precision:trans={n:v for n,v in trans.items() if design['specs'][n]['budget']==config['precision_audit']['budget']}
    draws=config['precision_audit']['draws'] if precision else config['check_information_draws']
    result=paired_task_information(exp,trans,n_samples=draws,seed=seed(config,'precision' if precision else 'check_mi',key),
        batch_size=config['information_batch_size'],return_sample_arrays=True)
    folder='precision' if precision else 'integration'
    save_arrays(OUT/folder/f'{key}.npz',result)
    if not precision:save_experiment(OUT/'inputs'/f'{key}.npz',exp)
    full=info_bits(result['full']);arrays=result['sample_arrays'];rows=[];decisions=[]
    for name in trans:
        bits=info_bits(result['representations'][name]);spec=design['specs'][name]
        loss=full-bits;prediction=design['predicted_full_bits']-design['predicted'][name]
        # Joint two-component numerical interval; no unbudgeted intersection
        # of95% intervals. True ratio is bounded by data processing.
        full_joint=summarize_bounded(arrays['full']['information'],(0.,1.),multiplicity=2)
        repr_joint=summarize_bounded(arrays['representations'][name]['information'],(0.,1.),multiplicity=2)
        denominator_lower=full_joint['confidence_lower']
        eligible=denominator_lower>=config['retention_fraction_min_full_information_bits']
        ratio_interval=None if not eligible else [max(0.,repr_joint['confidence_lower']/full_joint['confidence_upper']),
            min(1.,repr_joint['confidence_upper']/denominator_lower)]
        rows.append({'case_key':key,'bank':bank['name'],'task':task,'noise':regime_name,'domain':domain,'candidate':name,**spec,
            'sensor_bits':full,'representation_bits':bits,'loss_bits':loss,'predicted_loss_bits':prediction,
            'absolute_prediction_error_bits':abs(prediction-loss),
            'sensor_relative_retention':bits/full if eligible else None,
            'retention_fraction_eligible':eligible,'sensor_information_denominator_lower_bits':denominator_lower,
            'retention_fraction_joint_numerical_ci95':ratio_interval,
            'ratio_interval_scope':'Joint Bonferroni Hoeffding for full and represented MI; conditional on finite bank; unresolved below.05bit sensor lower endpoint',
            'sensor_information':result['full'],'known_state_reference':exp.oracle_summary(),
            'representation_information':result['representations'][name],'paired_loss':result['losses'][name]})
    if not precision:
        actual={n:info_bits(result['representations'][n]) for n in trans}
        for choice in design['selections']:
            names=[n for n in trans if design['specs'][n]['budget']==choice['budget']]
            oracle=best(names,actual);chosen=choice['candidate']
            pair=arrays['representations'][oracle]['information']-arrays['representations'][chosen]['information']
            pool_summaries=[summarize_bounded(arrays['representations'][n]['information']-arrays['representations'][chosen]['information'],
                (-1.,1.),multiplicity=len(names)**2) for n in names]
            regret_summary={'estimate':actual[oracle]-actual[chosen],
                'confidence_lower':max(0.,max(s['confidence_lower'] for s in pool_summaries)),
                'confidence_upper':max(0.,max(s['confidence_upper'] for s in pool_summaries)),
                'empirical_bernstein_lower':max(0.,max(s['empirical_bernstein_lower'] for s in pool_summaries)),
                'empirical_bernstein_upper':max(0.,max(s['empirical_bernstein_upper'] for s in pool_summaries)),
                'multiplicity':len(names)**2,'n_outer_draws':draws,
                'scope':'Max over all simultaneous candidate-minus-selected intervals; numerical conditional finite-pool regret'}
            decision={**choice,'case_key':key,'bank':bank['name'],'task':task,'noise':regime_name,'domain':domain,
                'selected_bits':actual[chosen],'best_candidate':oracle,'best_candidate_bits':actual[oracle],
                'point_regret_bits':actual[oracle]-actual[chosen],
                'regret_numerical_summary':regret_summary,
                'scope':'Best-in-pool point estimate is retrospective; simultaneous paired numerical interval accounts for all pool comparisons'}
            decisions.append(decision)
    metadata={'case_key':key,'bank':bank['name'],'task':task,'noise':regime_name,'domain':domain,
              'states':state['parameters'],'experiment':exp.metadata,'integration':result['metadata']}
    return rows,decisions,metadata

def sensor_atlas(banks,config):
    rows=[]
    for bank_name,bank in banks.items():
        counts=[19,64,256] if bank['gains_v_per_am'].shape[1]==256 else [19,64]
        for task in config['tasks']:
            for amplitude in [config['signal_peak_am'],config['sensitivity_signal_peak_am']]:
                state=source_means(bank,task,'check_id',config,amplitude)
                for noise,regime in config['noise_regimes'].items():
                    for channels in counts:
                        exp=experiment(bank,state,regime,channels,config);oracle=exp.oracle_summary()
                        row={'bank':bank_name,'task':task,'noise':noise,'channels':channels,'independent_sensor_time_dimensions':(channels-1)*config['n_times'],
                             'source_peak_am':amplitude,'known_state_reference':oracle,'known_state_bits':info_bits(oracle),'known_state_bayes_error':oracle_risk(oracle)}
                        # Hidden native/64 references on two independent spatial/temporal tasks.
                        if amplitude==config['signal_peak_am'] and task in config['decoder_tasks'] and noise in config['representation_noise_regimes']:
                            result=paired_task_information(exp,{},n_samples=config['check_information_draws'],
                                seed=seed(config,'atlas_mi',bank_name,task,noise,channels),batch_size=config['information_batch_size'])
                            row['hidden_state_information']=result['full'];row['hidden_state_bits']=info_bits(result['full'])
                        rows.append(row)
        print('Sensor atlas completed '+bank_name,flush=True)
    return rows

def apply_transform(epochs,transform):
    rs,tb=transform;return np.einsum('ac,nct,bt->nab',rs,epochs,tb,optimize=True).reshape(len(epochs),-1)

def prediction_metrics(labels,p,oracle_p,heads):
    p=np.clip(p,1e-6,1-1e-6);positive=(labels+1)/2
    predicted=np.where(p>=.5,1,-1);optimal=np.where(oracle_p>=.5,1,-1)
    errors=(predicted!=labels).astype(float);optimal_errors=(optimal!=labels).astype(float)
    logloss=-(positive*np.log2(p)+(1-positive)*np.log2(1-p))
    rows=[]
    for h in np.unique(heads):
        select=heads==h
        rows.append({'head_index':int(h),'n_epochs':int(select.sum()),'error':float(errors[select].mean()),
                     'bayes_classifier_realized_error':float(optimal_errors[select].mean()),
                     'paired_error_excess':float((errors-optimal_errors)[select].mean()),
                     'log_loss_bits':float(logloss[select].mean())})
    values=np.array([r['paired_error_excess'] for r in rows]);mean=float(values.mean())
    half=1.96*float(values.std(ddof=1)/np.sqrt(len(values)))
    accessible_summary=summarize_bounded(1-logloss,(-math.log2(1e6)+1,1.))
    # Decoder evaluation fixes head and label strata. Hoeffding applies to
    # independent nonidentical noise draws; iid sample-variance EB does not.
    for key in ['empirical_bernstein_lower','empirical_bernstein_upper','empirical_bernstein_halfwidth']:
        accessible_summary.pop(key,None)
    accessible_summary['interval_scope']='Hoeffding for independent stratified epoch draws conditional on fitted predictor and fixed finite states; empirical SE descriptive'
    return {'error':float(errors.mean()),'bayes_classifier_realized_error':float(optimal_errors.mean()),
        'paired_error_excess':float((errors-optimal_errors).mean()),'approximate_head_unit_ci95_error_excess':[mean-half,mean+half],
        'head_interval_scope':'Approximate across-state diagnostic; states within one anatomical model are not independent human subjects',
        'log_loss_bits':float(logloss.mean()),'accessible_information_lower_bound_estimate_bits':float(1-logloss.mean()),
        'accessible_information_numerical_summary':accessible_summary,
        'lower_bound_scope':'Clipped held-out predictor crossentropy lower bound on true task MI; point estimate is not exact MI',
        'clipping_probability':1e-6,'per_head':rows}

def decoder_study(banks,designs,config):
    rows=[]
    for bank_name,bank in banks.items():
        for task in config['decoder_tasks']:
            regime_name=config['decoder_noise_regime'];regime=config['noise_regimes'][regime_name]
            design=designs[f'{bank_name}__{task}__{regime_name}']
            train_exp=experiment(bank,source_means(bank,task,'development',config),regime,19,config)
            valid_exp=experiment(bank,source_means(bank,task,'validation',config),regime,19,config)
            def balanced_draw(exp,per_head,stage,domain=''):
                heads=np.repeat(np.arange(len(exp.plus)),per_head)
                labels=np.tile(np.r_[np.ones(per_head//2),-np.ones(per_head-per_head//2)],len(exp.plus))
                return draw_raw_epochs(exp,len(heads),seed(config,stage,bank_name,task,domain),head_indices=heads,labels=labels)
            train=balanced_draw(train_exp,config['decoder_training_epochs_per_head'],'decoder_train')
            valid=balanced_draw(valid_exp,config['decoder_validation_epochs_per_head'],'decoder_valid')
            full=(np.eye(18),np.eye(config['n_times']));choices={'full':full}
            choices['fixed_generic_pca']=design['transforms'][candidate_name('generic_pca',4,16,64)]
            for method in ['task_pca','task_fisher']:
                choice=next(c['candidate'] for c in design['selections'] if c['budget']==64 and c['policy']=='development_'+method)
                choices['predicted_'+method]=design['transforms'][choice]
            checks={}
            for domain in ['check_id','check_shift']:
                exp=experiment(bank,source_means(bank,task,domain,config),regime,19,config)
                checks[domain]=(exp,balanced_draw(exp,config['decoder_check_epochs_per_head'],'decoder_check',domain))
            for feature_name,transform in choices.items():
                xt=apply_transform(train.epochs,transform);xv=apply_transform(valid.epochs,transform)
                fits=[]
                for ridge in config['decoder_ridges']:
                    model=train_linear(xt,train.labels,ridge=ridge)
                    error=float(np.mean(model.predict(xv)!=valid.labels));fits.append((error,ridge,model))
                linear=min(fits,key=lambda row:(row[0],row[1]))[2]
                fits=[]
                for penalty in config['decoder_mlp_l2']:
                    model=train_mlp(xt,train.labels,xv,valid.labels,hidden_units=config['decoder_mlp_hidden_units'],
                        epochs=config['decoder_mlp_max_epochs'],patience=config['decoder_mlp_patience'],l2=penalty,
                        seed=seed(config,'mlp',bank_name,task,feature_name,penalty))
                    error=float(np.mean(model.predict(xv)!=valid.labels));fits.append((error,penalty,model))
                nonlinear=min(fits,key=lambda row:(row[0],row[1]))[2]
                for kind,model in [('linear',linear),('mlp',nonlinear)]:
                    name=f'{bank_name}__{task}__{feature_name}__{kind}'
                    params={key:value for key,value in vars(model).items() if key!='metadata'}
                    np.savez_compressed(OUT/'models'/f'{name}.npz',**params)
                    for domain,(check_exp,draw) in checks.items():
                        projected=prepare_transform_experiment(check_exp,*transform,name=feature_name)
                        projected_epochs=np.einsum('ac,nct,bt->nab',*([transform[0],draw.epochs,transform[1]]),optimize=True)
                        p=model.predict_proba(apply_transform(draw.epochs,transform));oracle_p=projected.posterior_raw(projected_epochs)
                        metrics=prediction_metrics(draw.labels,p,oracle_p,draw.head_indices)
                        rows.append({'bank':bank_name,'task':task,'noise':regime_name,'domain':domain,'features':feature_name,
                            'independent_features':xt.shape[1],'decoder':kind,'fit':model.metadata,**metrics})
                        np.savez_compressed(OUT/'models'/f'{name}__{domain}__predictions.npz',
                            labels=draw.labels,heads=draw.head_indices,probabilities=p,bayes_probabilities=oracle_p)
                print(f'Decoders completed {bank_name}/{task}/{feature_name}',flush=True)
    return rows

def summarize_results(rows,decisions,transfer,atlas,decoder,precision,config):
    summary={'prediction':[],'selection':[],'transfer':[],'sensor_atlas':[],'learning':[], 'precision':[]}
    for bank in config['geometry_banks']:
        for domain in ['check_id','check_shift']:
            selected=[r for r in rows if r['bank']==bank and r['domain']==domain and r['budget']==64]
            if not selected:continue
            errors=np.array([r['absolute_prediction_error_bits'] for r in selected])
            summary['prediction'].append({'bank':bank,'domain':domain,'n_candidate_conditions':len(selected),
                'mean_absolute_error_bits':float(errors.mean()),'median_absolute_error_bits':float(np.median(errors)),
                'p90_absolute_error_bits':float(np.quantile(errors,.9)),'criterion_met':bool(errors.mean()<=config['primary_forecast_mae_max_bits'])})
            for policy in ['development_information','mean_prototype_information','variance_retention','fixed_generic_pca']:
                chosen=[r for r in decisions if r['bank']==bank and r['domain']==domain and r['budget']==64 and r['policy']==policy]
                summary['selection'].append({'bank':bank,'domain':domain,'policy':policy,'n_cases':len(chosen),
                    'mean_regret_bits':float(np.mean([r['point_regret_bits'] for r in chosen])),
                    'mean_selected_bits':float(np.mean([r['selected_bits'] for r in chosen])),
                    'criterion_met':bool(np.mean([r['point_regret_bits'] for r in chosen])<=config['primary_selection_mean_regret_max_bits'])})
    for domain in ['check_id','check_shift']:
        for policy in ['development_information','mean_prototype_information','variance_retention','fixed_generic_pca']:
            chosen=[r for r in transfer if r['domain']==domain and r['policy']==policy]
            summary['transfer'].append({'domain':domain,'policy':policy,'n_cases':len(chosen),
                'mean_regret_bits':float(np.mean([r['point_regret_bits'] for r in chosen])),'mean_selected_bits':float(np.mean([r['selected_bits'] for r in chosen]))})
    for bank in config['geometry_banks']:
        for noise in config['noise_regimes']:
            ar=[r for r in atlas if r['bank']==bank and r['noise']==noise and r['channels']==19 and r['source_peak_am']==config['signal_peak_am']]
            summary['sensor_atlas'].append({'bank':bank,'noise':noise,'mean_known_state_bits':float(np.mean([r['known_state_bits'] for r in ar]))})
    for feature in config['decoder_features']:
        for kind in ['linear','mlp']:
            ar=[r for r in decoder if r['features']==feature and r['decoder']==kind]
            summary['learning'].append({'features':feature,'decoder':kind,'mean_error':float(np.mean([r['error'] for r in ar])),
                'mean_paired_error_excess':float(np.mean([r['paired_error_excess'] for r in ar]))})
    for p in precision:
        key=p['case_key'];coarse=next(r for r in rows if r['case_key']==key and r['candidate']==p['candidate'])
        summary['precision'].append({'case_key':key,'candidate':p['candidate'],'coarse_bits':coarse['representation_bits'],
            'precision_bits':p['representation_bits'],'absolute_difference_bits':abs(coarse['representation_bits']-p['representation_bits'])})
    return summary

def main():
    if (OUT/'run_manifest.json').exists():raise RuntimeError('Completed benchmark is immutable; use a new version for new runs')
    OUT.mkdir(parents=True,exist_ok=True)
    if any((OUT/d).exists() and any((OUT/d).iterdir()) for d in ['designs','integration','inputs','models']):
        raise RuntimeError('Partial attempt exists; archive it explicitly before rerun')
    for d in ['designs','integration','inputs','models','precision','figures']:(OUT/d).mkdir(exist_ok=True)
    config=json.loads(CONFIG.read_text());start=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    sources=[CONFIG,ROOT/'CEILING_BENCHMARK_PROTOCOL.md',Path(__file__),ROOT/'src/tdo_sim/ceiling_benchmark.py']
    sources+=list((ROOT/'src/tdo_sim').glob('*.py'))
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    dump(OUT/'pre_outcome_freeze.json',{'frozen_at_utc':started,'source_sha256':source_hashes,'config_sha256':sha(CONFIG),
        'science_frozen':True,'checked_outcomes_before_this_stage':False})
    banks=load_banks(config);designs,predictions=stage_designs(banks,config)
    rows=[];decisions=[];metadata=[];precision=[];transfer_rows=[];transfer_decisions=[]
    for bank_name,bank in banks.items():
        for task in config['tasks']:
            for noise in config['representation_noise_regimes']:
                design=designs[f'{bank_name}__{task}__{noise}']
                for domain in ['check_id','check_shift']:
                    r,d,m=check_family(bank,task,domain,noise,design,config);rows+=r;decisions+=d;metadata.append(m)
                    print('Check family completed '+m['case_key'],flush=True)
                if task in config['precision_audit']['tasks'] and noise==config['precision_audit']['noise_regime']:
                    for domain in config['precision_audit']['domains']:
                        r,_,_=check_family(bank,task,domain,noise,design,config,precision=True);precision+=r
    transfer=config['cross_anatomy_transfer'];sample=banks[transfer['check_bank']]
    for task in config['tasks']:
        design=designs[f'{transfer["development_bank"]}__{task}__{transfer["noise_regime"]}']
        # The same physical named19 contacts use one shared coverage order.
        if not np.array_equal(sample['montage_order'][:19],banks[transfer['development_bank']]['montage_order'][:19]):raise RuntimeError('Cross-anatomy montage mismatch')
        reduced={**design,'transforms':{n:v for n,v in design['transforms'].items() if design['specs'][n]['budget']==64},
                 'selections':[s for s in design['selections'] if s['budget']==64]}
        for domain in ['check_id','check_shift']:
            key=f'transfer__{task}__{transfer["noise_regime"]}__{domain}'
            r,d,m=check_family(sample,task,domain,transfer['noise_regime'],reduced,config,key_override=key)
            transfer_rows+=r;transfer_decisions+=d;metadata.append(m)
    atlas=sensor_atlas(banks,config);decoder=decoder_study(banks,designs,config)
    summary=summarize_results(rows,decisions,transfer_decisions,atlas,decoder,precision,config)
    for name,value in [('representation_conditions',rows),('decisions',decisions),('case_metadata',metadata),('anatomy_transfer_conditions',transfer_rows),
                       ('anatomy_transfer_decisions',transfer_decisions),('sensor_atlas',atlas),('decoder_conditions',decoder),('precision_conditions',precision),('summary',summary)]:
        dump(OUT/(name+'.json'),value)
    current={name:sha(ROOT/name) for name in source_hashes}
    if current!=source_hashes:raise RuntimeError('Source mutated during scientific run')
    dump(OUT/'run_manifest.json',{'protocol':config['protocol'],'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds':time.perf_counter()-start,'config_sha256':sha(CONFIG),'source_sha256_before':source_hashes,'source_sha256_after':current,
        'physical_assets_manifest_sha256':sha(ASSETS/'manifest.json'),'development_stage_manifest_sha256':sha(OUT/'development_stage_manifest.json'),
        'counts':{'representation_rows':len(rows),'decision_rows':len(decisions),'anatomy_transfer_rows':len(transfer_rows),
                  'anatomy_transfer_decisions':len(transfer_decisions),'sensor_atlas_rows':len(atlas),'decoder_rows':len(decoder),'precision_rows':len(precision)},
        'scope':config['side_information'],'packages':{'numpy':np.__version__},
        'output_sha256':{str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file()}})
    print('Ceiling benchmark completed '+json.dumps(clean(summary['prediction'])),flush=True)

if __name__=='__main__':main()
