"""Readable report of the fixed relation-loss experiment; no model selection."""
import html
import importlib.util
import json
from pathlib import Path
import shutil
from datetime import datetime

import pandas as pd

W=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('relation_experiment',W/'run.py')
exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
LABELS={'original_h':'原始 h','existing_raw20':'原始20轮 SAE','continue_mse':'MSE 继续训练','relation_weak':'弱关系约束 λ=0.03（主候选）','relation_strong':'强关系约束 λ=0.3（次要候选）'}


def result_table(rows,variant):
    out='<table><thead><tr><th>方法</th><th>Macro-F1</th><th>相对原始h</th><th>改善供体</th><th>描述性95%区间（pp）</th><th>seed均值变化（pp）</th></tr></thead><tbody>'
    for method in LABELS:
        r=next((r for r in rows if r['method']==method and r['variant']==variant),None)
        if r is None and method=='original_h':r=next(r for r in rows if r['method']==method)
        if r is None:continue
        seed=' / '.join(f"{s['seed']}: {s['delta_pp']:+.3f}" for s in r['seed_deltas']) or '确定性基线'
        low,high=r['descriptive_ci95_pp']
        out+=f"<tr><td>{LABELS[method]}</td><td>{r['macro_f1_percent']:.4f}%</td><td>{r['delta_pp']:+.4f} pp</td><td>{r['positive_donors']}/6</td><td>[{low:+.3f}, {high:+.3f}]</td><td>{seed}</td></tr>"
    return out+'</tbody></table>'


def main():
    rows=exp.read(exp.O/'summary.json');checks=exp.read(exp.O/'verification.json')
    assert checks['status']=='PASS'
    gate=exp.read(exp.O/'advance_gate.json')
    training=exp.read(exp.O/'training_summary.json')
    auditpath=exp.O/'EXPERIMENT_AUDIT.json';claimpath=exp.O/'CLAIMS_FROM_RESULTS.json'
    reviewed=auditpath.exists() and claimpath.exists()
    review='独立审查尚未完成；数值已完成确定性复算。'
    claim=''
    if reviewed:
        audit=exp.read(auditpath);c=exp.read(claimpath)
        review='独立上下文、同系列模型审查（provisional）：'+str(audit.get('overall_verdict',audit.get('verdict','unknown')))+" · <a href='EXPERIMENT_AUDIT.md'>审查记录</a> · <a href='CLAIMS_FROM_RESULTS.md'>结论边界</a>"
        claim='<p>'+html.escape(str(c.get('suggested_claim_revision','')))+'</p>'
    signal='主候选达到预设的描述性推进门槛，尚不构成独立数据确认。' if gate['pass_gate'] else '主候选未达到预设推进门槛。'
    contrast=pd.DataFrame(exp.read(exp.O/'planned_contrasts.json'))
    display=contrast[['variant','left','right','delta_pp','positive_donors']].copy()
    display['left']=display.left.map(LABELS);display['right']=display.right.map(LABELS)
    donor=pd.read_csv(exp.O/'paired_donor_metrics.csv')
    primary=donor[(donor.variant=='reconstruction')|(donor.method=='original_h')].pivot(index='test_donor',columns='method',values='delta_pp').rename(columns=LABELS)
    perclass=pd.read_csv(exp.O/'per_class_summary.csv')
    perclass=perclass[(perclass.variant=='reconstruction')|(perclass.method=='original_h')].pivot(index='cell_type',columns='method',values='delta_pp').rename(columns=LABELS)
    curves=[]
    for r in training:
        a,b=r['monitor_history'][0],r['monitor_history'][-1]
        curves.append(dict(method=LABELS[r['arm']],train_MSE_start=a['train_raw_mse'],train_MSE_final=b['train_raw_mse'],valid_MSE_start=a['valid_raw_mse'],valid_MSE_final=b['valid_raw_mse'],train_KL_start=a['train_monitor_kl'],train_KL_final=b['train_monitor_kl'],valid_train_KL_start=a['valid_to_train_kl'],valid_train_KL_final=b['valid_to_train_kl']))
    curve=pd.DataFrame(curves).groupby('method',sort=False).mean()
    curve.to_csv(exp.O/'learning_curve_summary.csv')
    grad=[]
    for r in training:
        for g in r['gradient_diagnostics']:
            grad.append(dict(fold=r['fold'],seed=r['seed'],method=r['arm'],step=g['step'],weighted_gradient_ratio=g['weighted_gradient_ratio'],normalized_mse=g['normalized_mse'],kl=g['kl']))
    pd.DataFrame(grad).to_csv(exp.O/'gradient_diagnostics.csv',index=False)
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAE MSE与关系保持实验</title>
<style>body{{font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:#f5f7fb;color:#24364a;margin:0}}main{{max-width:1400px;margin:auto;padding:30px}}section{{background:white;border:1px solid #dfe6ef;border-radius:12px;padding:22px;margin:20px 0;overflow:auto}}h1{{font-size:27px}}h2{{font-size:21px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #e1e7f0;text-align:left;white-space:nowrap}}th{{background:#f0f4fa}}a{{color:#245fc4}}.note{{color:#5c6a7b}}code{{background:#edf1f8;padding:3px 6px}}</style>
<main><h1>Kang：MSE 与 SAE 关系保持对照</h1><p class='note'>{datetime.now().isoformat(timespec='seconds')} · scGPT 冻结 · 原维度512 · 历史探索性六供体</p>
<section><h2>预设主问题</h2><p>{signal}</p><p>弱关系约束能否同时优于原始h与等预算MSE继续训练？强约束和固定一半混合是预先声明的次要分析，不能按TEST成绩升级为主方法。</p><p>{review}</p>{claim}</section>
<section><h2>主结果：直接重建512维，不搜索混合系数</h2>{result_table(rows,'reconstruction')}<p>固定训练结束step1000的checkpoint；只用VALID标签选择同一11档Ridge正则化。三个SAE seed先在供体内平均，再对六供体等权平均。</p></section>
<section><h2>预定成对比较</h2>{display.to_html(index=False,float_format=lambda x:f'{x:+.4f}',border=0)}<p>主比较是弱关系重建对原始h、对MSE继续训练。全部区间仅为描述性供体bootstrap，六折训练集相互重叠；不能按细胞或18次fit计算独立样本显著性。</p></section>
<section><h2>固定一半混合：次要结果</h2><p><code>h′ = 0.5h + 0.5重建</code>；系数提前固定，不做alpha选择或原始h回退。</p>{result_table(rows,'fixed_half')}</section>
<section><h2>逐供体主结果变化（pp）</h2>{primary.to_html(float_format=lambda x:f'{x:+.3f}',border=0)}</section>
<section><h2>逐细胞类型主结果变化（pp）</h2>{perclass.to_html(float_format=lambda x:f'{x:+.3f}',border=0)}<p>固定八类Macro-F1，测试供体缺失的类别计0；另保存仅存在类别的Macro-F1。Megakaryocytes等稀有类别完整保留。</p></section>
<section><h2>训练是否实际改变了表征</h2>{curve.to_html(float_format=lambda x:f'{x:.6g}',border=0)}<p>上表均为每折/seed监控值的等权平均。TRAIN KL使用固定供体均衡批次；VALID KL使用验证查询与固定TRAIN锚点。它们是自监督几何诊断，不能视为真实生物关系或分类成绩。</p><p>总计54次继续训练，每次1000次更新；监控5个训练状态，梯度检查5次。每组从相同checkpoint出发，重置Adam，使用同一批次索引与余弦学习率。合计训练耗时{sum(r['seconds'] for r in training)/60:.2f}分钟，不含读出、I/O及核验。</p><p>MSE继续训练相对原20轮的比较同时包含额外更新、供体均衡采样、优化器重置和学习率变化，不能将其差异唯一归因于训练轮数。</p></section>
<section><h2>实现与核验</h2><p><code>L = normalized_MSE + λ × mean_query KL(P_original || P_reconstructed)</code>。TRAIN均值中心化后计算cosine，温度0.1；同供体和自身不作为训练锚点，每个查询192个跨供体候选。原始教师停止梯度，log_softmax保证概率归一化，KL使用float64。</p><p>{checks['prediction_models']}份选中读出/测试预测完成标签、指标、显式einsum及独立Cholesky复算；{checks['training_runs']}份训练记录和配对批次核验；{checks['embedding_files_read_back']}份512维embedding重新读取通过。</p><p>冻结骨干、SAE训练及关系目标均不使用细胞类型标签；VALID标签仍用于后续Ridge选择，因此整个评价流程不应称为完全无标签。新字典已更新，旧F编号的生物学归因不能自动沿用。</p></section>
<section><h2>证据范围与文件</h2><p>当前数据已用于历史方法探索，不是新盲测。保持原h的关系不增加新的观测信息，也不保证改善分类；跨供体相似性不等于真实细胞类型邻居。PBMC与SDCV特征筛选未运行。</p><p><a href='summary.csv'>指标CSV</a> · <a href='planned_contrasts.json'>成对比较</a> · <a href='PROTOCOL.json'>固定协议</a> · <a href='PILOT.json'>pilot</a> · <a href='verification.json'>核验</a> · <a href='embedding_manifest.json'>实际512维向量索引</a> · <a href='REPRODUCIBILITY.txt'>复现说明</a></p></section></main></html>'''
    (exp.O/'REPORT.html').write_text(text)
    (exp.O/'REPRODUCIBILITY.txt').write_text('''冻结scGPT；历史Kang ctrl9577细胞/6供体/8类，512维最后层有符号均值embedding。每折四TRAIN、一VALID、一TEST。全部新选择锁定后统一测试；这是历史数据复用的探索性结果。

运行环境 /opt/anaconda3/envs/scgpt/bin/python，torch2.3.1，CPU4线程。
入口：
python work/sae_relation/run.py preflight
python work/sae_relation/run.py pilot
python work/sae_relation/run.py develop
python work/sae_relation/run.py lock
python work/sae_relation/run.py evaluate
python work/sae_relation/verify.py
python work/sae_relation/report.py

协议PROTOCOL.json；来源哈希work/sae_relation/sources.json；选型锁SELECTION_LOCK.json；batch索引与模型/Adam恢复文件在work/sae_relation/。进度checkpoint每250次更新保存；完成后删除临时优化器恢复文件，保留最终网络。
SAE全部为2048宽、raw-value TopK32、无decoder bias、单位范数decoder列；从对应旧checkpoint继续训练1000步，所有组配对批次、初始化和lr。
主输出为直接重建；固定一半混合为次要输出，未搜索混合系数、未回退到h、未按TEST选择lambda或checkpoint。
embedding_manifest.json中每份向量仅含该折留出供体，含原始rows、labels和donor；并非全数据拟合同一个SAE。
''')
    dest=exp.O/'reproducibility';dest.mkdir(exist_ok=True)
    for name in ['run.py','verify.py','report.py']:shutil.copy2(W/name,dest/name)
    shutil.copy2(exp.ROOT/'work/sae_refine/run.py',dest/'frozen_sae_runtime.py')
    shutil.copy2(exp.ROOT/'work/sae_complementarity/core.py',dest/'frozen_readout_core.py')
    exp.save(exp.O/'EVIDENCE_PRECHECK.json',dict(status='verified',sources={str(p):exp.old.sha(p) for p in [exp.O/'all_metrics.json',exp.O/'summary.json',exp.O/'verification.json',exp.O/'advance_gate.json']},meaning='deterministic existence and metric checks; semantic interpretation still needs review'))
    tracker=exp.read(exp.O/'EXPERIMENT_TRACKER.json');tracker.update(status='COMPLETE' if reviewed else 'VERIFIED_PENDING_REVIEW',report='complete',review='same-family / provisional' if reviewed else 'pending')
    exp.save(exp.O/'EXPERIMENT_TRACKER.json',tracker)
    print('REPORT_WRITTEN',exp.O/'REPORT.html',flush=True)


if __name__=='__main__':main()
