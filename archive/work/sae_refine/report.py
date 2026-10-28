"""Standalone human-readable report; no new parameter selection."""
from datetime import datetime
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as exp

LABELS={
    'original_h':'原始 embedding',
    'single_raw':'单个 TopK SAE（2048 / K32）',
    'ensemble_existing3':'已有 3 个 SAE 集成（总宽6144 / 总K96）',
    'wide6144_k96':'单个大 SAE（6144 / K96）',
    'abs2048_k32':'AbsTopK SAE（2048 / K32）',
    'abs_sparse_z':'AbsTopK 稀疏 z（2048维，诊断）',
    'bag2_1024_k16':'两个小 SAE 平均（总宽2048 / 总K32）',
    'boost2_1024_k16':'两级残差 Boosting（总宽2048 / 总K32）',
}


def table(rows,variant):
    out='<table><thead><tr><th>方法</th><th>Macro-F1</th><th>相对原 h</th><th>改善供体</th><th>选中原 h</th><th>配对 seed 变化</th></tr></thead><tbody>'
    for r in rows:
        if r['variant']!=variant:
            continue
        if variant=='forced_reconstruction' and r['method']=='original_h':
            continue
        color='positive' if r['delta_pp']>0 else 'negative'
        seed=' / '.join(f"{s['seed']}: {s['delta_pp']:+.3f}" for s in r['seed_deltas']) or '单个确定性/集成结果'
        out+=f"<tr><td>{LABELS[r['method']]}</td><td>{r['macro_f1_percent']:.4f}%</td><td class='{color}'>{r['delta_pp']:+.4f} pp</td><td>{r['positive_donors']}/6</td><td>{r['identity_selected']}/{r['fits']}</td><td>{seed}</td></tr>"
    return out+'</tbody></table>'


def contrasts(records):
    donor=pd.read_csv(exp.O/'paired_donor_metrics.csv')
    out=[]
    for variant in ['mixed512','forced_reconstruction']:
        for left,right in [('ensemble_existing3','wide6144_k96'),('abs2048_k32','single_raw'),('bag2_1024_k16','single_raw'),('boost2_1024_k16','single_raw'),('boost2_1024_k16','bag2_1024_k16')]:
            a=donor[(donor.method==left)&(donor.variant==variant)].set_index('fold').macro_f1
            b=donor[(donor.method==right)&(donor.variant==variant)].set_index('fold').macro_f1
            diff=(a-b)*100
            out.append(dict(variant=variant,left=left,right=right,delta_pp=float(diff.mean()),positive_donors=int((diff>1e-10).sum()),negative_donors=int((diff< -1e-10).sum()),donor_differences_pp=diff.tolist()))
    exp.save(exp.O/'capacity_contrasts.json',out)
    lines=[]
    for c in out:
        lines.append(f"<tr><td>{c['variant']}</td><td>{LABELS[c['left']]} − {LABELS[c['right']]}</td><td>{c['delta_pp']:+.4f} pp</td><td>{c['positive_donors']}/6</td></tr>")
    return '<table><tr><th>表示</th><th>匹配比较</th><th>平均变化</th><th>正向供体</th></tr>'+''.join(lines)+'</table>'


def main():
    rows=exp.read(exp.O/'summary.json')
    verify=exp.read(exp.O/'verification.json')
    assert verify['status']=='PASS'
    training=exp.read(exp.O/'training_summary.json')
    baseline=next(r for r in rows if r['method']=='original_h' and r['variant']=='mixed512')
    mixed=[r for r in rows if r['variant']=='mixed512' and r['method']!='original_h']
    qualified=[LABELS[r['method']] for r in mixed if r['advance_gate']]
    signal='、'.join(qualified)+' 达到预设扩大验证门槛。' if qualified else '本轮没有方法达到预设扩大验证门槛（平均+0.5 pp、至少4/6供体改善，适用时至少2/3 seed均值为正）。'
    contrast=contrasts(rows)
    donor=pd.read_csv(exp.O/'paired_donor_metrics.csv')
    donor=donor[donor.variant=='mixed512'].pivot(index='test_donor',columns='method',values='delta_pp')
    donor=donor.rename(columns=LABELS)
    classes=pd.read_csv(exp.O/'per_class_summary.csv')
    classes=classes[classes.variant=='mixed512'].pivot(index='cell_type',columns='method',values='delta_pp').rename(columns=LABELS)
    review='审核结果尚未写入；数值已完成确定性复算。'
    if (exp.O/'EXPERIMENT_AUDIT.json').exists():
        audit=exp.read(exp.O/'EXPERIMENT_AUDIT.json')
        review='独立上下文、同系列模型审核（provisional）：'+str(audit.get('overall_verdict',audit.get('verdict','unknown')))+" · <a href='EXPERIMENT_AUDIT.md'>完整审查</a> · <a href='CLAIMS_FROM_RESULTS.md'>结论边界</a>"
    claim=''
    if (exp.O/'CLAIMS_FROM_RESULTS.json').exists():
        c=exp.read(exp.O/'CLAIMS_FROM_RESULTS.json')
        claim='<p>'+html.escape(str(c.get('suggested_claim_revision',c.get('summary',''))))+'</p>'
    selection_note=''
    if (exp.O/'selection_diagnostics.json').exists():
        diagnostics=exp.read(exp.O/'selection_diagnostics.json')
        ens=next(r for r in diagnostics['methods'] if r['method']=='ensemble_existing3')
        selection_note=f"<p>三成员集成：混合参数选择相对固定重建在VALID平均增加{ens['mean_valid_mixed_minus_forced_pp']:.3f} pp，在TEST平均变化{ens['mean_test_mixed_minus_forced_pp']:+.3f} pp。选中α分布为{html.escape(str(ens['mix_alpha_histogram']))}。这是已选模型的事后描述，不能据此用TEST重新选择α，也不能单独确定失败原因。<a href='selection_diagnostics.json'>选择诊断</a></p>"
    numerical_note=''
    if (exp.O/'NUMERICAL_NOTES.json').exists():
        n=exp.read(exp.O/'NUMERICAL_NOTES.json')
        numerical_note=f"<p class='note'>参考库路径曾发出NumPy运行时警告，原因本轮未确定；六折sklearn参考得分最大误差{n['sklearn_comparison_max_abs_error']:.2e}，全部保存得分与显式einsum复算最大误差{n['primary_vs_explicit_einsum_max_abs_error']:.2e}。主拟合使用SciPy BLAS。详见<a href='NUMERICAL_NOTES.json'>数值记录</a>。</p>"
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kang SAE refinement 实验</title><style>
body{{font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#213547;background:#f5f7fb;margin:0}}main{{max-width:1380px;margin:auto;padding:32px}}section{{background:white;border:1px solid #e0e6ef;border-radius:12px;padding:24px;margin:20px 0;overflow:auto}}h1{{font-size:28px}}h2{{font-size:21px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;border-bottom:1px solid #e5e9f0;text-align:left;white-space:nowrap}}th{{background:#f1f4fa}}.positive{{color:#087a5a}}.negative{{color:#a14735}}.note{{color:#596778}}code{{background:#f0f3f8;padding:3px 5px}}a{{color:#245ec4}}
</style><main><h1>Kang：冻结 scGPT 的 SAE 表征优化</h1><p class="note">生成时间 {datetime.now().isoformat(timespec='seconds')} · 探索性历史数据复用 · 所有主结果保持512维</p>
<section><h2>结果边界</h2><p>{signal}</p><p>原始 h：{baseline['macro_f1_percent']:.4f}% Macro-F1。六供体轮换，每次四供体训练、一供体验证、一供体测试；三个训练 seed；已有三成员集成每折只有一个集成结果。</p><p>{review}</p>{claim}<p>同一验证供体用于无监督 checkpoint 与下游参数选择。测试集在全部验证选择锁定后统一评价，但这些供体已用于历史方法探索，不能称为全新盲测。</p></section>
<section><h2>主结果：验证集选择的混合512维表示</h2><p><code>h′ = h + α(重建 − h)</code>；α ∈ {{0, .25, .5, .75, 1}}。身份候选优先打破平局。各供体先平均 seed，再对六供体等权平均。</p>{table(rows,'mixed512')}<p class="note">seed变化单位为pp。原始 h 确定性计算；三个 SAE 成员与三个 seed 不是独立供体。</p></section>
<section><h2>强制修改：α固定为1，只选择Ridge正则化</h2><p>本表检验重建表示本身，不允许回退到原 h。</p>{table(rows,'forced_reconstruction')}{selection_note}</section>
<section><h2>稀疏编码诊断</h2>{table(rows,'z_only')}<p>z-only 是2048维分类读出，不属于512维 embedding 改善的主结果。</p></section>
<section><h2>容量与结构对照</h2>{contrast}<p>容量匹配指总字典宽度与TopK槽位，以及最大字典加权训练轮数；early stopping、有效激活和实际墙钟耗时可能不同。Bagging取平均，Boosting求和，这是方法本身的区别。</p></section>
<section><h2>逐供体变化（pp）</h2>{donor.to_html(float_format=lambda x:f'{x:+.3f}',border=0)}<p>六供体训练集相互重叠；summary.json中的供体bootstrap区间仅作描述，不能按细胞或18次fit计算独立样本置信度。</p></section>
<section><h2>逐细胞类型变化（pp）</h2>{classes.to_html(float_format=lambda x:f'{x:+.3f}',border=0)}<p>固定八类Macro-F1；某测试供体缺失的类别仍计0，另行保存仅存在类别的Macro-F1。稀有类别完整保留。</p></section>
<section><h2>训练与核验</h2><p>{len(training)}个训练记录（包含1个原流程复现），合计训练耗时{sum(t['seconds'] for t in training)/60:.2f}分钟，不含分类、读取缓存、核验与导出。原流程复现的全部checkpoint张量误差为0。</p><p>{verify['prediction_files_verified']}份测试预测逐项复算；{verify['training_runs_verified']}份训练记录核验；{verify['test_embedding_files']}份512维测试embedding导出。原始 Ridge 在六折均与 sklearn 独立求解对照。</p>{numerical_note}<p>DIAL式技术方向删除尚未运行：本轮没有建立可安全移除的技术专属方向；这不是一个已运行阴性结果。PBMC未参与本轮拟合或测试。</p></section>
<section><h2>可复用文件</h2><p><a href="summary.csv">主指标CSV</a> · <a href="paired_donor_metrics.csv">逐供体指标</a> · <a href="per_class_summary.csv">逐类别汇总</a> · <a href="PROTOCOL.json">固定方案</a> · <a href="verification.json">确定性核验</a> · <a href="embedding_manifest.json">512维embedding索引</a> · <a href="REPRODUCIBILITY.txt">复现说明</a></p></section></main></html>'''
    (exp.O/'REPORT.html').write_text(body)
    desc='''本轮是已探索Kang数据上的开发比较，非新盲测。scGPT冻结，所有新SAE仅使用各折四个训练供体；验证供体用于MSE checkpoint及分类超参数选择。

运行环境：/opt/anaconda3/envs/scgpt/bin/python，torch2.3.1，CPU4线程。
固定协议：PROTOCOL.json。划分和原始行编号：SPLITS.json。选型锁：SELECTION_LOCK.json。

复现入口（保留原始缓存路径，见脚本与source manifest）：
python work/sae_refine/run.py preflight
python work/sae_refine/run.py develop
python work/sae_refine/run.py lock
python work/sae_refine/run.py evaluate
python work/sae_refine/verify.py
python work/sae_refine/report.py

新训练checkpoint与线性读出保存在work/sae_refine/，不覆盖任何旧实验。公开输出含全部测试预测、逐供体/类别指标、选中的512维测试embedding。
embedding_manifest.json列出的每份embedding仅含该折留出测试供体，带原始rows和labels；不同fold的SAE由不同训练供体拟合，不能称为全数据同一个SAE。
identity为true意味着验证选中了alpha=0，应视为原始embedding回退。forced_reconstruction是alpha=1的独立验证选型。
三seed集成在每折只有1个集成模型；不将成员或seed作为独立供体。
最大预算容量匹配，实际early stopping训练量见training_summary.json。
'''
    (exp.O/'REPRODUCIBILITY.txt').write_text(desc)
    dest=exp.O/'reproducibility'
    dest.mkdir(exist_ok=True)
    for name in ['run.py','verify.py','report.py']:
        shutil.copy2(exp.W/name,dest/name)
    shutil.copy2(exp.ROOT/'work/sae_complementarity/core.py',dest/'frozen_readout_core.py')
    exp.save(exp.O/'EVIDENCE_PRECHECK.json',{'status':'verified','source_files':{str(p):exp.old.sha(p) for p in [exp.O/'summary.json',exp.O/'all_metrics.json',exp.O/'verification.json']},'note':'deterministic source existence and metric recomputation; not semantic claim acceptance'})
    tracker=exp.read(exp.O/'EXPERIMENT_TRACKER.json')
    review_done=(exp.O/'EXPERIMENT_AUDIT.json').exists() and (exp.O/'CLAIMS_FROM_RESULTS.json').exists()
    tracker.update(status='COMPLETE' if review_done else 'VERIFIED_PENDING_REVIEW',verification='PASS',report='complete',review='same-family / provisional' if review_done else 'pending')
    exp.save(exp.O/'EXPERIMENT_TRACKER.json',tracker)
    print('REPORT_WRITTEN',exp.O/'REPORT.html',flush=True)


if __name__=='__main__':
    main()
