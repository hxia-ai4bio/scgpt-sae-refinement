from pathlib import Path
import hashlib,json,shutil
out=Path('outputs/sae_figures')
caption='''Figure 2 | SAE 特征与细胞身份及技术因素的关联。
使用冻结 scGPT 的 512 维 cell embedding、2048 维 TopK SAE，K=32。
A：固定解释性字典中，依据 TRAIN 供体表现选择的代表特征。颜色为各细胞类型供体等权平均激活在行内的 z-score，不能用于比较不同特征的绝对激活量。横轴给出 TRAIN 细胞数；仅有 7 个细胞的巨核细胞未用于候选特征选择。
B：B、CD14 单核与 NK 的留出供体单特征 AUROC。每点对应一个 Kang ctrl 留出供体，先在供体内平均三个种子，共六个供体；黑色短线为供体均值。每个折/种子分别在 TRAIN 中选择坐标，不能理解为同一 F 编号跨字典复现。
C：645 个满足覆盖要求的特征，其最强细胞类型关联与技术因素关联。纵轴为校正细胞类型和供体后的 QC 偏相关绝对值最大值；标出三个细胞类型候选和技术对照 F1724。
D：在两个 PBMC 批次中的探索性读出，展示 TRAIN 选择的各折/种子字典的平均 AUROC。两批次不作为供体重复；CD14 的外部表现接近随机，说明外部一致性因细胞类型而异。

Figure 3 | B 细胞相关特征 F160 的局部基因敏感性。
使用原始解释性 SAE 的固定字典（主折 test patient_1016、seed 42），并非后续表达重建优化后的 SAE。
A：主留出供体全部 1723 个细胞的 F160 激活，包括 125 个 B 细胞。箱体表示中位数和四分位距，须为第 5–95 百分位；散点每类最多抽取 50 个，仅用于显示，横轴数量对应全部细胞。
B：TRAIN 排名最高的 15 个正贡献基因。贡献为将一个输入基因的表达 bin 替换为参考 bin 后，F160 激活的下降量。误差线为四个 TRAIN 供体间标准差；采用 8 个 TRAIN 原型细胞，每供体 2 个。参考细胞先在原型内平均，再在供体内和供体间平均；对应输入中缺失的基因贡献记零。右侧为基因出现在输入中的原型覆盖数。灰色为 RPL/RPS 前缀基因。
C：依据 TRAIN 排名，在同一历史留出供体的 4 个选定 B 细胞中逐步替换基因 bin；与联合匹配随机基因对照比较。细线为单个原型，粗线为四个原型均值；随机对照先在每个原型内平均 10 组匹配集合及参考配对。横轴是请求替换的预算，实际覆盖受输入基因限制。预算 40 时平均激活下降为 0.248289，对照为 0.135295。这是模型输入替换敏感性，不是生物学基因敲除。所有已测替换中 F160 均保留在 TopK，其表现与对应稀疏化前编码器坐标一致，不能据此宣称稀疏化产生了该敏感性。
D：预先保留的 GO/Hallmark 候选注释，同时列出普通 ORA 和匹配随机检验的全局 q 值。全局检验族包含 6462 项检验，匹配随机全局 FDR<0.05 的发现为零，因此不能将这些注释写成已证实的 B 细胞功能模块。非 RPL/RPS 列表是排除这类前缀基因后的 top40。

数据与复现：本次仅重绘已有检查过的结果，没有新增训练、模型选择或显著性检验。面板来源数据保存在同目录 CSV 中，输入文件 SHA256 与数值核对见 FIGURE_DATA_AND_CHECKS.json。固定字典的特征编号不应与其他种子或优化后字典直接对应。输出包含 350 dpi PNG、矢量 PDF 和可编辑文字 SVG。
'''
(out/'FIGURE_CAPTIONS_ZH.txt').write_text(caption,encoding='utf-8')
shutil.copyfile('work/sae_figures/draw_figures.py',out/'draw_figures.py')
checks=json.loads((out/'FIGURE_DATA_AND_CHECKS.json').read_text())
for name,expected in checks['sources'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.iterdir()) if p.is_file() and p.name!='DELIVERY_VERIFICATION.json'}
for stem in ['Figure2_SAE_overview','Figure3_Bcell_F160']:
    assert (out/(stem+'.pdf')).read_bytes().startswith(b'%PDF')
    assert (out/(stem+'.png')).read_bytes().startswith(b'\x89PNG')
    assert '<svg' in (out/(stem+'.svg')).read_text()
(out/'DELIVERY_VERIFICATION.json').write_text(json.dumps({'all_15_input_hashes_unchanged':True,'visual_QA':'Both PNGs inspected; titles, axes, tables and panel labels readable without clipping.','files':files},indent=2))
print('Verified',len(files),'deliverables; all 15 input hashes unchanged')
