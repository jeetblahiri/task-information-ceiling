#!/usr/bin/env python3
"""Report and plot frozen ceiling results without changing scientific outputs.

All aggregates are equal-weight descriptions of the declared simulation grid.
Numerical intervals condition on that grid; none describe a human population.
"""
from __future__ import annotations
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/tdo_ceiling_matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/tdo_ceiling_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'results/ceiling_benchmark_v1'
SUP = ROOT / 'results/ceiling_benchmark_v1_supplements'
OUT = ROOT / 'results/ceiling_benchmark_v1_presentation'
BANKS = ['sphere_rep1', 'sphere_rep2', 'bem_fsaverage', 'bem_sample']
TASKS = ['spatial_lateral', 'spatial_nearby', 'spatial_front_back', 'temporal_early_late', 'temporal_fine_latency', 'temporal_rhythm']
NAMES = dict(zip(TASKS, ['Lateral patches', 'Nearby patches', 'Front/back patches', 'Early/late event', '12-ms latency', '8/24-Hz waveform']))
BNAMES = dict(zip(BANKS, ['Sphere bank 1', 'Sphere bank 2', 'Template BEM', 'Individual BEM']))
COLORS = dict(zip(BANKS, ['#2875B0', '#62A8A5', '#E59B38', '#BB5966']))
METHODS = ['generic_pca', 'task_pca', 'task_fisher']
MCOLORS = ['#9A9FA5', '#247BB2', '#CE6543']
MLABELS = ['Generic pooled PCA', 'Task-matched PCA', 'Task Fisher']
DOMAINS = ['check_id', 'check_shift']
INPUTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def savefig(fig, stem):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT/'figures'/f'{stem}.{ext}', dpi=200, facecolor='white')
    plt.close(fig)


def csv_rows(name, rows):
    with (OUT/(name+'.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def pair_summary(decisions, domain, transfer=False):
    """Independent stratified Hoeffding: two domain means jointly at95%."""
    families = sorted({r['case_key'] for r in decisions if r['domain']==domain and r['budget']==64})
    values = []
    for family in families:
        choices = {r['policy']:r['candidate'] for r in decisions if r['case_key']==family and r['budget']==64}
        path = MAIN/'integration'/f'{family}.npz'
        INPUTS[str(path.relative_to(ROOT))] = sha(path)
        with np.load(path) as f:
            values.append(f[choices['fixed_generic_pca']+'__information']-f[choices['development_information']+'__information'])
    x = np.concatenate(values)
    half = 2*np.sqrt(np.log(2*2/.05)/(2*len(x)))
    return {'domain':domain, 'n_cases':len(families), 'n_independent_integration_draws':len(x),
        'fixed_generic_minus_development_bits':float(x.mean()),
        'joint_numerical_ci95':[float(x.mean()-half), float(x.mean()+half)],
        'scope':'Two-domain Bonferroni Hoeffding, independent finite-case integration draws; no anatomy-population claim'}


def figures(atlas, reps, transfer, tdec, decoders, cal):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'svg.fonttype':'none', 'pdf.fonttype':42})
    fig, axes = plt.subplots(2,2,figsize=(13,8.8))
    noises = ['iid','spatial','temporal','joint']
    short = ['IID','Corr','AR1','Both']
    for ax,bank in zip(axes.flat,BANKS):
        ns = [19,64,256] if bank.startswith('sphere') else [19,64]
        cells = [[next(r['known_state_bits'] for r in atlas if (r['bank'],r['task'],r['noise'],r['channels'],r['source_peak_am'])==(bank,task,noise,ch,2e-8)) for ch in ns for noise in noises] for task in TASKS]
        matrix = np.asarray(cells)
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        for (i,j),v in np.ndenumerate(matrix):
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if v<.5 else '#20272E')
        ax.set_yticks(range(6),[NAMES[t] for t in TASKS],fontsize=9)
        ax.set_xticks(range(matrix.shape[1]),[f'{ch}\n{s}' for ch in ns for s in short],fontsize=8)
        for x in range(4,matrix.shape[1],4):ax.axvline(x-.5,color='white',linewidth=1.7)
        ax.set_title(BNAMES[bank]+(' (native 256)' if bank.startswith('sphere') else ' (native 64)'),loc='left',weight='bold')
    fig.subplots_adjust(left=.15,right=.91,bottom=.13,top=.90,hspace=.34,wspace=.42)
    cax=fig.add_axes([.93,.2,.014,.6]);fig.colorbar(im,cax=cax,label='State-known task information (bits)')
    fig.suptitle('1  Acquisition sets a task-conditioned ceiling',x=.02,ha='left',fontsize=16,weight='bold')
    fig.text(.02,.035,'Balanced binary targets; 250 ms; nominal source scale 20 nAm; marginal noise SD fixed at 6 µV.\nEntries average 24 declared states, with prescribed source amplitude variation. Entire state revealed: optimistic references.',fontsize=10)
    savefig(fig,'01_sensor_information_atlas')

    fig,axes=plt.subplots(2,3,figsize=(13,8.5))
    for ax,task in zip(axes.flat,TASKS):
        pool=[r for r in reps if r['task']==task and r['noise']=='joint' and r['domain']=='check_id' and r['budget']==64]
        for method,color,label in zip(METHODS,MCOLORS,MLABELS):
            y=[np.mean([r['representation_bits'] for r in pool if r['method']==method and r['spatial_features']==s]) for s in [1,2,4,8]]
            ax.plot(range(4),y,'o-',color=color,label=label,linewidth=2)
        full=np.mean([r['sensor_bits'] for r in pool]);ax.axhline(full,ls='--',color='#273542',label='Full sensors')
        ax.set_xticks(range(4),['1 × 64','2 × 32','4 × 16','8 × 8'],fontsize=9)
        ax.set_ylim(bottom=0,top=max(full*1.13,.045));ax.grid(alpha=.18)
        ax.set_title(NAMES[task],loc='left',weight='bold');ax.set_ylabel('Retained information (bits)')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.53,.935),frameon=False)
    fig.subplots_adjust(left=.07,right=.98,bottom=.16,top=.84,wspace=.30,hspace=.36)
    fig.suptitle('2  The same feature budget permits different information loss',x=.02,ha='left',fontsize=16,weight='bold')
    fig.text(.02,.045,'X axis: spatial directions × contiguous temporal averages, always 64 scalar features (19 contacts).\nEqual-weight means over four banks; ordinary check domain, joint correlated/colored noise. Designs use development states only.\nPoint estimates describe this grid; scalar features are not foundation-model tokens.',fontsize=10)
    savefig(fig,'02_fixed_budget_retention')

    fig,axes=plt.subplots(2,2,figsize=(13,9.6))
    for ax,domain in zip(axes[0],DOMAINS):
        for bank in BANKS:
            a=[r for r in reps if r['bank']==bank and r['domain']==domain and r['budget']==64]
            ax.scatter([r['predicted_loss_bits'] for r in a],[r['loss_bits'] for r in a],s=10,alpha=.55,color=COLORS[bank],label=BNAMES[bank])
        limit=max(max(r['loss_bits'],r['predicted_loss_bits']) for r in reps if r['budget']==64 and r['domain']==domain)*1.04
        ax.plot([0,limit],[0,limit],color='#333333',ls='--',lw=1)
        ax.set_xlim(-.01,limit);ax.set_ylim(-.01,limit)
        mae=np.mean([r['absolute_prediction_error_bits'] for r in reps if r['budget']==64 and r['domain']==domain])
        ax.set_title(('Ordinary' if domain=='check_id' else 'Broader source shift')+f': within-family MAE {mae:.4f} bits',loc='left',weight='bold')
        ax.set_xlabel('Development-predicted loss (bits)');ax.set_ylabel('Untouched-check loss (bits)');ax.grid(alpha=.17)
    axes[0,0].legend(frameon=False,fontsize=8,loc='upper left')
    for ax,domain in zip(axes[1],DOMAINS):
        x=np.arange(6);width=.23
        full=[];generic=[];selected=[];known=[];err=[]
        for task in TASKS:
            choices={r['policy']:r for r in tdec if r['task']==task and r['domain']==domain}
            chosen=next(r for r in transfer if r['task']==task and r['domain']==domain and r['candidate']==choices['development_information']['candidate'])
            full.append(chosen['sensor_bits']);generic.append(choices['fixed_generic_pca']['selected_bits']);selected.append(chosen['representation_bits'])
            known.append(chosen['representation_information']['known_head_exact']['average_information_bits'])
        ax.bar(x-width,full,width,color='#749ABD',label='Full observation')
        ax.bar(x,generic,width,color='#DEB454',label='Fixed generic PCA')
        ax.bar(x+width,selected,width,color='#BB5966',label='Template task choice')
        ax.scatter(x+width,known,marker='D',s=20,edgecolor='#222222',facecolor='white',zorder=5,label='Its state-known ceiling')
        ax.set_xticks(x,['Lateral','Nearby','Front/back','Early/late','12-ms','8/24-Hz'],rotation=18,fontsize=9)
        ax.set_ylabel('Individual-anatomy information (bits)');ax.set_ylim(0,.20);ax.grid(axis='y',alpha=.17)
        ax.set_title(('Ordinary' if domain=='check_id' else 'Shifted')+' states: blind template → individual transfer',loc='left',weight='bold')
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.52,.09),frameon=False,fontsize=9)
    fig.subplots_adjust(left=.07,right=.98,bottom=.22,top=.90,wspace=.25,hspace=.42)
    fig.suptitle('3  Accurate local forecasts do not imply portable lossy encoders',x=.02,ha='left',fontsize=16,weight='bold')
    fig.text(.02,.03,'Top: all 64-feature candidates, two noise regimes; untouched check states within each forward family.\nBottom: matrices frozen on template only; 19 contacts, joint noise. Diamonds bound expected post-encoding calibration\nunder the stated fixed-encoder, independent-epoch model. Bars are numerical estimates; intervals are in the saved tables.',fontsize=10)
    savefig(fig,'03_prediction_and_anatomy_transfer')

    fig,axes=plt.subplots(2,2,figsize=(13,9.0))
    features=['full','fixed_generic_pca','predicted_task_pca','predicted_task_fisher']
    featurelabels=['Full (1152)','Generic (64)','Task PCA (64)','Fisher (64)']
    x=np.arange(4);w=.24
    oracle=[]
    for feature in features:
        a=[r for r in decoders if r['features']==feature and r['decoder']=='linear']
        oracle.append(np.mean([r['bayes_classifier_realized_error'] for r in a]))
    axes[0,0].bar(x-w,100*np.array(oracle),w,color='#A6ADB4',label='Matched Bayes (realized)')
    for j,(model,color,label) in enumerate([('linear','#2875B0','Ridge LDA'),('mlp','#BB5966','24-unit MLP')]):
        errors=[100*np.mean([r['error'] for r in decoders if r['features']==f and r['decoder']==model]) for f in features]
        axes[0,0].bar(x+j*w,errors,w,color=color,label=label)
        excess=[100*np.mean([r['paired_error_excess'] for r in decoders if r['features']==f and r['decoder']==model]) for f in features]
        axes[0,1].bar(x+(j-.5)*.3,excess,.3,color=color,label=label)
    for ax in axes[0]:ax.set_xticks(x,featurelabels,fontsize=9);ax.grid(axis='y',alpha=.17)
    axes[0,0].set_ylabel('Error (%)');axes[0,0].set_ylim(0,43);axes[0,0].legend(frameon=False,fontsize=8)
    axes[0,0].set_title('Observed error and matched representation reference',loc='left',weight='bold')
    axes[0,1].set_ylabel('Paired excess error (percentage points)');axes[0,1].set_title('Finite learning / parameter / shift gap',loc='left',weight='bold')
    for ax,task in zip(axes[1],['spatial_lateral','temporal_fine_latency']):
        for bank in BANKS:
            a=sorted([r for r in cal if r['bank']==bank and r['task']==task and r['domain']=='check_shift'],key=lambda r:r['calibration_epochs'])
            y=np.array([r['full']['bits'] for r in a]);lo=np.array([r['full']['confidence_lower_bits'] for r in a]);hi=np.array([r['full']['confidence_upper_bits'] for r in a])
            ax.errorbar(range(4),y,yerr=[y-lo,hi-y],fmt='o-',color=COLORS[bank],lw=1.6,capsize=2,label=BNAMES[bank])
            ax.axhline(a[0]['known_head_exact']['average_information_bits'],color=COLORS[bank],ls=':',alpha=.6)
        ax.set_xticks(range(4),[0,2,8,32]);ax.set_xlabel('Same-subject known-label calibration epochs')
        ax.set_ylabel('Full-observation conditional information (bits)');ax.set_ylim(bottom=0);ax.grid(alpha=.17)
        ax.set_title(NAMES[task]+': true-prior calibration, shifted states',loc='left',weight='bold')
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.52,.09),frameon=False,fontsize=9)
    fig.subplots_adjust(left=.07,right=.98,bottom=.18,top=.90,wspace=.28,hspace=.36)
    fig.suptitle('4  Measurement, representation and learning require separate comparisons',x=.02,ha='left',fontsize=15,weight='bold')
    fig.text(.02,.04,'Top: equal-weight means over 16 bank × task × domain conditions; fixed finite training, no foundation model.\nBottom: full-test / full-calibration finite-prior references; fresh noises, persistent state; pointwise 95% numerical Hoeffding bars.\nDotted lines reveal the entire state. Operational development-prior calibration results are reported separately.',fontsize=10)
    savefig(fig,'04_learning_and_calibration')


def main():
    refresh='--presentation-refresh' in sys.argv
    if (OUT/'manifest.json').exists() and not refresh:
        raise RuntimeError('Presentation already finalized; use explicit presentation-refresh with retained revision evidence')
    if refresh and not (ROOT/'evidence/ceiling_presentation_before_revision.json').exists():
        raise RuntimeError('Refresh requires a retained presentation revision record')
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    config=read(ROOT/'config/ceiling_benchmark.json')
    manifest=read(MAIN/'run_manifest.json');suppmanifest=read(SUP/'manifest.json')
    atlas=read(MAIN/'sensor_atlas.json');reps=read(MAIN/'representation_conditions.json');decisions=read(MAIN/'decisions.json')
    transfer=read(MAIN/'anatomy_transfer_conditions.json');tdec=read(MAIN/'anatomy_transfer_decisions.json')
    decoders=read(MAIN/'decoder_conditions.json');summary=read(MAIN/'summary.json')
    cal=read(SUP/'calibration_information.json');op=read(SUP/'calibration_operational.json');mesh=read(SUP/'mesh_sensitivity.json');geometry=read(SUP/'geometry_checks.json')
    resolution=read(ROOT/'results/ceiling_benchmark_v1_transfer_resolution/conditions.json')
    read(ROOT/'results/ceiling_benchmark_v1_transfer_resolution/manifest.json')
    audit=read(ROOT/'evidence/ceiling_benchmark_integrity_checks.json')
    assert audit['passed'] and not audit['failures'] and not audit['numerical_inequality_flags']
    figures(atlas,reps,transfer,tdec,decoders,cal)
    pair_transfer=[pair_summary(tdec,d,True) for d in DOMAINS]
    pair_local=[pair_summary(decisions,d) for d in DOMAINS]
    transfer_table=[]
    for domain in DOMAINS:
        for task in TASKS:
            choices={r['policy']:r for r in tdec if r['task']==task and r['domain']==domain}
            row=next(r for r in transfer if r['task']==task and r['domain']==domain and r['candidate']==choices['development_information']['candidate'])
            known=row['representation_information']['known_head_exact']
            delta=np.asarray(known['separations'])**2/np.maximum(np.asarray(row['known_state_reference']['separations'])**2,1e-300)
            transfer_table.append(dict(domain=domain,task=task,candidate=row['candidate'],sensor_bits=row['sensor_bits'],
                fixed_generic_bits=choices['fixed_generic_pca']['selected_bits'],selected_bits=row['representation_bits'],
                selected_state_known_bits=known['average_information_bits'],selected_bayes_error=row['representation_information']['bayes_error'],
                state_known_squared_separation_retention_median=float(np.median(delta))))
    calibration_table=[]
    for row in cal:
        operational=next(r for r in op if (r['bank'],r['task'],r['domain'],r['calibration_epochs'])==(row['bank'],row['task'],row['domain'],row['calibration_epochs']))
        gain=row['paired_calibration_information_gain']
        calibration_table.append(dict(bank=row['bank'],task=row['task'],domain=row['domain'],calibration_epochs=row['calibration_epochs'],
            conditional_bits=row['full']['bits'],paired_gain_bits=gain['estimate'],paired_gain_lower=gain['confidence_lower'],paired_gain_upper=gain['confidence_upper'],
            state_known_bits=row['known_head_exact']['average_information_bits'],operational_error=operational['error']['estimate'],operational_log_loss_bits=operational['log_loss_bits']['estimate']))
    noise_table=[dict(bank=bank,**{noise:float(np.mean([r['known_state_bits'] for r in atlas if r['bank']==bank and r['channels']==19 and r['noise']==noise and r['source_peak_am']==2e-8])) for noise in ['iid','spatial','temporal','joint']}) for bank in BANKS]
    csv_rows('anatomy_transfer',transfer_table);csv_rows('calibration',calibration_table);csv_rows('noise_factorial',noise_table)
    resolution_table=[dict(task=r['task'],domain=r['domain'],mesh=r['mesh'],full_state_known_bits=r['full_state_known']['average_information_bits'],
        template_choice_state_known_bits=r['representations']['development_information']['average_information_bits'],
        fixed_generic_state_known_bits=r['representations']['fixed_generic_pca']['average_information_bits']) for r in resolution]
    csv_rows('transfer_resolution_sensitivity',resolution_table)
    csv_rows('predictions',summary['prediction']);csv_rows('selections',summary['selection']);csv_rows('learning',summary['learning'])
    flat=[{k:v for k,v in r.items() if not isinstance(v,(list,dict)) and k!='retention_fraction_joint_numerical_ci95'} for r in reps]
    csv_rows('representation_conditions',flat)
    statistics=dict(prediction=summary['prediction'],selection=summary['selection'],transfer=summary['transfer'],
        learning=summary['learning'],noise=noise_table,anatomy_transfer=transfer_table,calibration=calibration_table,
        transfer_paired_grid_comparisons=pair_transfer,within_family_paired_grid_comparisons=pair_local,
        post_outcome_transfer_resolution_sensitivity=resolution_table,
        maximum_precision_difference_bits=max(r['absolute_difference_bits'] for r in summary['precision']),
        mesh=[dict(bank=b,relative_gain_change=geometry[b]['ico3_to_ico4_relative_car_gain'],
            mean_absolute_information_change_bits=float(np.mean([abs(r['difference_bits']) for r in mesh if r['bank']==b])),
            maximum_absolute_mean_information_change_bits=max(abs(r['difference_bits']) for r in mesh if r['bank']==b)) for b in ['bem_fsaverage','bem_sample']])
    dump(OUT/'statistics.json',statistics)
    forecasts=table(['Bank','Check domain','MAE (bits)','90th percentile (bits)','≤0.05 criterion'],[[BNAMES[r['bank']],r['domain'],f"{r['mean_absolute_error_bits']:.4f}",f"{r['p90_absolute_error_bits']:.4f}",r['criterion_met']] for r in summary['prediction']])
    policyrows=[]
    for policy in ['development_information','mean_prototype_information','variance_retention','fixed_generic_pca']:
        pool=[r for r in decisions if r['budget']==64 and r['policy']==policy]
        policyrows.append([policy,f"{np.mean([r['point_regret_bits'] for r in pool]):.5f}",len(pool)])
    policies=table(['Frozen policy','Mean regret (bits)','Check cases'],policyrows)
    transfers=table(['Task','Full sensor bits','Fixed generic bits','Template task-choice bits','Its state-known ceiling','Its Bayes error'],[[NAMES[r['task']],f"{r['sensor_bits']:.4f}",f"{r['fixed_generic_bits']:.4f}",f"{r['selected_bits']:.4f}",f"{r['selected_state_known_bits']:.4f}",f"{100*r['selected_bayes_error']:.1f}%"] for r in transfer_table if r['domain']=='check_id'])
    noises=table(['Bank','IID','Spatial correlation','Temporal color','Both'],[[BNAMES[r['bank']]]+[f"{r[n]:.4f}" for n in ['iid','spatial','temporal','joint']] for r in noise_table])
    learner=table(['Input','Decoder','Mean error','Excess over matched Bayes'],[[r['features'],r['decoder'],f"{100*r['mean_error']:.2f}%",f"{100*r['mean_paired_error_excess']:.2f} percentage points"] for r in summary['learning']])
    calibration=table(['Bank / shifted target','Cal0 bits','Cal32 bits','Paired gain (95% numerical interval)','Operational cal0→32 error'],[[BNAMES[b]+' / '+NAMES[t],f"{next(r['conditional_bits'] for r in calibration_table if (r['bank'],r['task'],r['domain'],r['calibration_epochs'])==(b,t,'check_shift',0)):.4f}",f"{next(r['conditional_bits'] for r in calibration_table if (r['bank'],r['task'],r['domain'],r['calibration_epochs'])==(b,t,'check_shift',32)):.4f}",f"{next(r['paired_gain_bits'] for r in calibration_table if (r['bank'],r['task'],r['domain'],r['calibration_epochs'])==(b,t,'check_shift',32)):.4f} ± 0.0212",f"{100*next(r['operational_error'] for r in calibration_table if (r['bank'],r['task'],r['domain'],r['calibration_epochs'])==(b,t,'check_shift',0)):.2f}% → {100*next(r['operational_error'] for r in calibration_table if (r['bank'],r['task'],r['domain'],r['calibration_epochs'])==(b,t,'check_shift',32)):.2f}%"] for b in BANKS for t in ['spatial_lateral','temporal_fine_latency']])
    mesh_table=table(['BEM family','Relative CAR gain difference','Mean absolute MI change','Maximum mean MI change'],[[BNAMES[r['bank']],f"{100*r['relative_gain_change']:.2f}%",f"{r['mean_absolute_information_change_bits']:.4f} bits",f"{r['maximum_absolute_mean_information_change_bits']:.4f} bits"] for r in statistics['mesh']])
    transfer_resolution=table(['Target / finer individual BEM','Full state-known bits','Template-choice state-known bits','Generic state-known bits'],[[NAMES[r['task']],f"{r['full_state_known_bits']:.4f}",f"{r['template_choice_state_known_bits']:.4f}",f"{r['fixed_generic_state_known_bits']:.4f}"] for r in resolution_table if r['domain']=='check_id' and r['mesh']=='ico4'])
    report=f'''# Task-conditioned EEG information ceilings: completed benchmark

Completed locally on 8 October 2026. This is a new simulation study implementing [INFORMATION_CEILING_SCOPE.md](../INFORMATION_CEILING_SCOPE.md), separate from all earlier studies and their failed method gates. Main run: {manifest['elapsed_seconds']:.1f} s; physical preparation: 66.2 s; supplementary sensitivity/calibration: {suppmanifest['elapsed_seconds']:.1f} s. No raw EEG, compression-project research or pretrained model was used. The study now owns a private environment containing hash-verified copies of installed third-party libraries, with no other project's research imports or package-version changes. Public anatomical assets are attributed in [provenance](../resources/ceiling_anatomy/provenance.json).

**Result:** information loss was prospectively predictable within the four declared forward families. A lossy representation selected using one anatomical model transferred poorly to the other, even under the correct evaluation-law oracle. This separates destruction by encoding from failure of a fitted predictor. Calibration can resolve retained-state ambiguity, but cannot exceed the fixed encoder's state-known information under the stated epoch-independence contract. These are conditional simulation insights using established theory, not a universal EEG capacity or demonstrated foundation-model result.

## 1. What was frozen and executed

The [protocol](../CEILING_BENCHMARK_PROTOCOL.md) and [configuration](../config/ceiling_benchmark.json) fixed six positive, non-antipodal binary distinctions, two fresh spherical banks, two BEM geometries, noise/source scales, split seeds, matrices/policies, feature budgets and usefulness criteria before new check outcomes. Each bank has8 development,4 validation and24 check states:144 physical operators total. The two BEM families are a template and one public individual anatomy, each with cap perturbations. They are **two geometries**, not72 independent human subjects. The nominal geometry, 32-source cortical-normal basis and source law remain material assumptions.

One observation is64 samples at256Hz (250ms). The nominal target source coefficient-vector peak scale is20nAm, multiplied by the persistent source-amplitude factor (±15% ordinary, ±30% shifted);8nAm is the nominal acquisition sensitivity. Both classes contain persistent background, and each epoch has an additional independent rank-one Gaussian source background. A subject's head, source parameters and static background persist across epochs. Sensor-noise marginal SD stays6µV while spatial correlation0/.55 and AR(1) color0/.75 vary independently. All processing propagates the physical covariance; it never renormalizes SNR after compression.

The primary representation comparison uses19 contacts,18 independent CAR coordinates and64 scalar features. Four spatial×temporal allocations (1×64,2×32,4×16,8×8) cross generic pooled PCA, task-matched PCA and approximate separable Fisher design. Temporal processing is ordinary contiguous averaging. Budgets32 and128 are retained as additional frozen tests. A scalar feature is not an EEG foundation-model token.

Executed:3,456 representation conditions,2,016 prospective decision rows,144 blind-transfer conditions,84 transfer choices,480 acquisition cells,128 fitted decoder conditions,96 independent high-precision conditions; supplementary48 BEM model/resolution cells,64 exact-prior calibration cells and64 operational calibration cells. Condition counts are not subject counts.

## 2. Matched information and accuracy references

Every binary target has entropy1bit. Reported MI is bits **per declared observation about that target**. Values for overlapping tasks cannot be summed into total EEG information. The benchmark does not maximize over source laws to estimate channel capacity. The oracle knows the specified finite latent-state law; zero-calibration mixture MI hides the actual state. Its known-state reference additionally reveals head/source/static state, an optimistic side-information change.

Within a known state, arbitrary class midpoints and a common known covariance permit the established Gaussian separation and error reference Phi(−d/2). Mixtures use their actual likelihoods instead. Exact likelihood reduction retains every affine class-mean and background direction, not merely the class contrast. Numerical-zero reductions are computational sufficient coordinates, separate from the lossy encoders being tested. [Mathematical audit](ceiling_benchmark_mathematics.md) records this distinction.

Full sensor and represented information share the same target/window/state prior and subject metadata. A fitted learner's gap to its matched representation oracle includes parameter estimation, finite training and distribution shift. Poor probe accuracy alone does not demonstrate absent information. Clipped held-out predictive cross-entropy supplies a lower-bound estimate, not exact representation MI. Ratio estimates are withheld when the simultaneous numerical sensor-information lower endpoint is below0.05bits.

## 3. Prediction works within the declared family

Development-only predictions were saved before check evaluation. The primary64-feature loss predictions use8 development states; the check conditions use24 untouched states, in ordinary and broader source-jitter/latency domains.

{forecasts}

All eight bank/domain **descriptive** MAE criteria pass. They do not certify generalization to a continuous human-anatomy population. The independent65,536-draw checks differed from the primary estimates by at most{statistics['maximum_precision_difference_bits']:.4f}bits among96 prespecified representation cells. Tiny differences between selection policies should therefore not be promoted into substantive superiority.

{policies}

All development-information mean-regret groups meet the frozen0.02-bit criterion; regret is relative to the12 frozen candidates at this budget, not all possible encoders. The mean-prototype Gaussian policy is nearly as good. The evidence supports useful model-informed allocation predictions, not the necessity or superiority of a complex uncertainty method. The information-selected encoder has lower same-family loss than the fixed generic baseline on average, but that finding does not carry across the held-out anatomical family.

![Fixed-budget retention](../results/ceiling_benchmark_v1_presentation/figures/02_fixed_budget_retention.png)

## 4. Blind anatomical-family transfer reveals a different limit

The template's encoders and selections were applied unchanged to the individual-anatomy check states. No individual check states were used to refit them. Tasks, montage names and covariance settings match; cortical positions/orientations and discrete source support follow each anatomical model. Thus this is a **forward/source-family transfer**, not an isolated causal intervention on head shape alone.

Ordinary-state outcomes at64 features:

{transfers}

Across all six tasks, the template information choice retains0.01168bits on ordinary individual states, versus0.06872bits for fixed generic4×16 PCA. Under broader source shift, the corresponding means are0.00770 and0.05297bits. For these two six-task grid means, the generic-minus-task advantages are respectively{pair_transfer[0]['fixed_generic_minus_development_bits']:.4f} and{pair_transfer[1]['fixed_generic_minus_development_bits']:.4f}bits; joint95% numerical Hoeffding intervals are [{pair_transfer[0]['joint_numerical_ci95'][0]:.4f},{pair_transfer[0]['joint_numerical_ci95'][1]:.4f}] and [{pair_transfer[1]['joint_numerical_ci95'][0]:.4f},{pair_transfer[1]['joint_numerical_ci95'][1]:.4f}]. These concern integration of the fixed six-task banks, not uncertainty across people.

The lateral contrast is a concrete example: full observation has0.17593bits; fixed generic features retain0.15966; the template task choice retains0.00665. Even revealing the whole state raises its encoded reference only to0.00720bits. Its Bayes error is46.17%, compared with29.92% for full observation. This is measured encoding loss under the declared law; it does not depend on a probe being too small or undertrained. The early/late, fine-latency and rhythm choices likewise have very low state-known encoded references. Nearby/front-back contrasts retain more signal, so the failure is task-specific.

The [calibration corollary](ceiling_calibration_corollary.md) gives I(Y;Z|C)≤I(Y;Z|H) for a **fixed encoder**, independent known-label calibration and fresh test noise conditional on the complete persistent state. It explains why post-encoding calibration cannot recover the omitted test coordinates in these examples. A calibration-adaptive encoder, raw-data re-encoding, longer context or extra modalities changes that contract. Current calibration curves evaluate full observations; they are not direct recovery experiments for the transferred encoders.

![Prediction and blind transfer](../results/ceiling_benchmark_v1_presentation/figures/03_prediction_and_anatomy_transfer.png)

## 5. Noise amplitude, acquisition and learning are separate effects

At19 contacts, these are six-task mean **state-known** information values at20nAm:

{noises}

Temporal color causes a larger drop than the tested spatial correlation in this grid, despite identical marginal noise amplitude. That ordering is a consequence of these waveform/covariance spectra, not a rule that correlation always reduces information. Sensor additions preserve existing recoverable differences under the model; the saved nested-montage table supplies task-specific gains rather than a universal channel threshold. For example, sample-BEM fine-latency state-known information in joint noise rises only0.0268→0.0433bits from19→64 contacts, whereas lateral information rises0.1842→0.2476bits.

![Acquisition map](../results/ceiling_benchmark_v1_presentation/figures/01_sensor_information_atlas.png)

Fitted decoders use1,536 training epochs (8×192),512 validation epochs (4×128) and3,072 check epochs (24×128) per task/domain. Training-only normalization and independent validation choose LDA ridge/MLP regularization. There is no foundation-model training or scaling experiment. Equal-weight results over16 bank×task×domain contexts:

{learner}

Full observations contain at least as much target information as their transforms, yet the finite learner can perform better on compression. Noise-aware compression improves this training regime's estimation problem while preserving useful task directions. The MLP does not consistently improve over the linear decoder. Neither finding establishes an asymptotic model-size ceiling. Per-head intervals are descriptive states within a model, not human-population confidence intervals.

## 6. Calibration and anatomical sensitivity constrain interpretation

Exact-prior calibration and operational prediction were kept distinct. For each outer draw, calibration and test share the same persistent state; their epoch noise/background amplitudes are fresh. Full calibration updates the true24-state check prior for the exact reference. The operational predictor instead uses only8 development prototypes on genuinely unseen states. Operational uncertainty uses256 independently drawn subject units, each with32 test epochs; it does not count8,192 dependent test epochs as independent subjects.

{calibration}

The paired gains compare calibrated/uncalibrated posteriors on the **same test draws**. Subtracting separately integrated cal0/cal32 means gives a different noisy estimate. Listed gain intervals are pointwise95% Hoeffding (±0.0212bits), not familywise intervals or human coverage claims. Some small point reversals across calibration budgets are compatible with numerical uncertainty. The development-prior predictor's error does not improve consistently with more calibration; exact information availability and successful operational use are separate. Its misspecified posterior entropy is never labelled true MI.

{mesh_table}

These compare matched nominal caps and source/static/epoch-background states at19 contacts over24 task/noise cells per anatomy. Template gain change is8.44%; individual gain change is23.17%. Individual mean information changes are at most0.0383bits across this grid. The supplied individual coarse FIF and native surfaces differ geometrically (nearest-vertex discrepancy up to5.74mm), so its comparison includes model/smoothing differences. Binary surface coordinates match independent official ASCII coordinates within6.1nm. The finer result is a model-plus-resolution sensitivity, **not a convergence proof or physical validation**. The main matched-ico3 outcomes were retained unchanged. The observed differences matter for the absolute low-information regimes, and prevent interpreting the displayed decimals as physiology.

An additional **post-outcome diagnostic** applies the already frozen template/generic encoders to both saved individual nominal-cap models, over all six tasks and both domains. It makes no new selection and uses state-known Gaussian references, with the same source/static/background laws. Ordinary-domain finer-model results:

{transfer_resolution}

The lateral encoded ceiling is0.00743bits on the coarse nominal cap and0.00749bits on the finer model, versus full-state references0.18427 and0.16813bits. All three temporal-task template choices remain below0.0012bits in the ordinary finer-model domain. Thus the prominent signal-loss mechanism survives this available sensitivity; it is not rescued by this finer model. This check remains restricted to nominal caps and the same sparse source law, and does not isolate pure discretization or prove population robustness. [Diagnostic manifest](../results/ceiling_benchmark_v1_transfer_resolution/manifest.json) and [independent verification](../evidence/ceiling_transfer_resolution_integrity_checks.json) retain its post-outcome status.

![Learning and calibration](../results/ceiling_benchmark_v1_presentation/figures/04_learning_and_calibration.png)

## 7. Checks, reproducibility and contribution scope

All50 core/protocol tests passed before outcomes. The independent [main verifier](../evidence/ceiling_benchmark_integrity_checks.json) reports zero implementation/hash failures and zero simultaneous numerical inequality flags. It checks574 saved output hashes,15 sources, physical inputs/outputs and preservation of every prior scientific manifest; recomputes9,600 known-state references, endpoint summaries, forecasts, prospective choices, regret intervals and fitted-model metrics. Eight actual-case1152-dimensional likelihood replays and16 dense64-dimensional encoder replays match within3.9×10⁻¹⁴. Separate Hoeffding and empirical-Bernstein intervals are never silently intersected. BEM sources, units, normals, cap surface positions, reference ranks, seeded splits and raw transform covariance are checked.

The [supplement audit](../evidence/ceiling_supplement_integrity_checks.json), [calibration integrand replay](../results/ceiling_benchmark_v1_calibration_replay/manifest.json) and [completion record](../evidence/ceiling_benchmark_completion.json) record final verification. Integration uncertainty applies to declared finite laws. No calibrated population certificate from earlier simpler models is automatically transferred to this richer experiment. Repairs, including the anatomical-file mismatch, are retained in [implementation ledger](../evidence/ceiling_implementation_log.json); no scientific targets, amplitudes, thresholds or selections were tuned to check results. The current mechanism explanation is post-outcome; the encoder, predictions and blind-transfer test were fixed beforehand.

The differentiated contribution is a **validated separation of sensor information, encoding loss, acquisition/source-family portability and finite learning performance**, with a practical counterexample to trusting locally optimal lossy features across anatomy. This is a candidate empirical insight paper using classical theory. Known comparisons already cover EEG information limits, prior-informed spatial design, robust detection, task-oriented placement and information-preserving projection; see the [prior comparison](insight_comparison_audit.md) and [literature audit](literature_audit.md). No first-discovery or unsolved-problem claim follows from these simulations.

For foundation-model design, the evidence motivates retaining spatial/temporal information until subject adaptation can occur, testing pre-encoder filtering under transfer, and evaluating model learning gaps against matched representations. It does not require adopting the tested Fisher algorithm. A publication-strength extension should test more independent anatomies and a denser cortical basis, and preferably one frozen EEG encoder with a validated information bound or matched Bayes/decoding reference. The present32-source simulation, two anatomical geometries, rank-one background, finite mixtures and chosen physical scales limit breadth. **The benchmark implementation is complete; Q1 acceptance or manuscript readiness is not established by computational completion alone.** A novel theorem remains optional.

## Reproduction and files

Use the local Python environment documented in the [implementation plan](../CEILING_IMPLEMENTATION_PLAN.md). Existing completed runners refuse to overwrite finalized runs. Main numerical results, calibration sensitivities and report outputs have separate manifests; report regeneration uses a separately named presentation version if changed. Machine-readable CSVs and full PNG/SVG/PDF exports are in [presentation outputs](../results/ceiling_benchmark_v1_presentation/). All printed table values are generated from frozen JSONs by [report script](../scripts/report_ceiling_benchmark.py).
'''
    for old,new in {
        'has8 development,4 validation and24 check':'has 8 development, 4 validation and 24 check',
        'states:144':'states: 144', 'not72':'not 72',
        'is64 samples at256Hz (250ms)':'is 64 samples at 256 Hz (250 ms)',
        'is20nAm;8nAm':'is 20 nAm; 8 nAm', ';8nAm':'; 8 nAm', 'stays6µV':'stays 6 µV',
        'correlation0/.55':'correlation 0/.55', 'color0/.75':'color 0/.75',
        'uses19 contacts,18 independent CAR coordinates and64 scalar':'uses 19 contacts, 18 independent CAR coordinates and 64 scalar',
        '(1×64,2×32,4×16,8×8)':'(1×64, 2×32, 4×16, 8×8)',
        'Budgets32 and128':'Budgets 32 and 128',
        'Executed:3,456 representation conditions,2,016 prospective decision rows,144 blind-transfer conditions,84 transfer choices,480 acquisition cells,128 fitted decoder conditions,96 independent high-precision conditions; supplementary48 BEM model/resolution cells,64 exact-prior calibration cells and64 operational calibration cells.':'Executed: 3,456 representation conditions, 2,016 prospective decision rows, 144 blind-transfer conditions, 84 transfer choices, 480 acquisition cells, 128 fitted decoder conditions, 96 independent high-precision conditions; supplementary 48 BEM model/resolution cells, 64 exact-prior calibration cells and 64 operational calibration cells.',
        'oracle knows the specified finite latent-state law':'oracle knows the specified finite latent-state law',
        'at64 features':'at 64 features', 'knows8 development':'knows 8 development',
        'only8 development':'only 8 development', 'true24-state':'true 24-state',
        'of16 bank':'of 16 bank', 'over16 bank':'over 16 bank',
        'uses1,536':'uses 1,536', 'epochs (8×192),512':'epochs (8×192), 512',
        '(4×128) and3,072':'(4×128) and 3,072', '256 independently':'256 independently',
        'with32 test':'with 32 test', 'count8,192':'count 8,192',
        'budget means':'budget means', 'checks574':'checks 574',
        'hashes,15':'hashes, 15', 'recomputes9,600':'recomputes 9,600',
        'actual-case1152':'actual-case 1152', 'and16 dense64':'and 16 dense 64',
        'over24 task':'over 24 task', 'two geometries':'two geometries',
        'present32-source':'present 32-source',
    }.items():report=report.replace(old,new)
    report=report.replace('](../results/ceiling_benchmark_v1_presentation/figures/',f']({OUT}/figures/')
    # Plain-language spacing in generated prose; never rewrite numbers, paths,
    # table cells or machine-readable scientific files.
    lines=[]
    for line in report.splitlines():
        if line and not line.startswith(('|','#','!','[')):
            chunks=re.split(r'(\[[^\]]*\]\([^)]*\))',line)
            for i,chunk in enumerate(chunks):
                if chunk.startswith('['):continue
                chunk=re.sub(r'(?<=[A-Za-z]):(?=[0-9])',': ',chunk)
                chunk=re.sub(r'\b(has|use|uses|is|at|not|entropy|among|all|All|below|supplementary|Budgets|independent|primary|the|reveals|their)(?=[0-9])',r'\1 ',chunk)
                chunk=re.sub(r'(?<=[0-9])(bits?|nAm|µV|ms|samples|contacts|states|Hz)\b',r' \1',chunk)
                chunks[i]=chunk
            line=''.join(chunks)
        lines.append(line)
    report='\n'.join(lines)+'\n'
    reportpath=ROOT/'reports/ceiling_benchmark_results.md';reportpath.write_text(report)
    output_files=[p for p in OUT.rglob('*') if p.is_file() and p!=OUT/'manifest.json']+[reportpath]
    dump(OUT/'manifest.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)),input_sha256=INPUTS,
        output_sha256={str(p.relative_to(ROOT)):sha(p) for p in output_files},
        scope='Read-only presentation and post-outcome mechanism summaries; all frozen scientific outcomes unchanged',
        figure_count=4,figure_formats=['PNG','SVG','PDF']))
    print('Created four figures, reproducible CSVs/statistics, report and presentation manifest')


if __name__=='__main__':main()
