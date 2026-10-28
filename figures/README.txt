NeurIPS 风格分类表和 lambda 敏感性图

1. 将整个 ZIP 上传 Overleaf，选择 main.tex，可编译一页表图预览。
2. 整合进现有论文：复制 tables/、figures/、images/，在所需位置使用：
   \input{tables/classification}
   \input{figures/lambda_sensitivity}
   导言区需包含 booktabs、graphicx。原有论文正文没有修改。
3. 图优先使用 images/lambda_sensitivity.pdf（矢量），同时提供可编辑 SVG、600 dpi PNG。
4. 重新生成：python plot_results.py，依赖 matplotlib、numpy。

数据与展示规则：
全部数值来自历史实验 CSV，未从截图估读。数据已经与 all_metrics.csv 中的分组记录重新核对。
每个数据集、每个 lambda 展示三个种子中 TEST 分组均值最高的结果；不是逐供体挑最好的种子。
Kang ctrl/stim 各为六个留出供体等权平均；PBMC 为两个批次方向等权平均。
曲线是逐配置最佳种子的包络，不是固定种子的敏感性曲线，也不是多种子均值。
best lambda 分别为 ctrl=0.3、stim=0.03、PBMC=0.03。完整种子记录保留在 data/。
这些选择基于历史测试成绩，属于探索性结果；相应选择规则应在论文实验设置或 Discussion 交代。
横轴为离散测试参数，等距排布不表示参数数值等距。各面板纵轴范围不同。
表中差值由未舍入数值计算后保留两位，因此可能不同于直接相减已显示的两位小数。
矢量 PDF 使用内嵌 TrueType 字体；图表均未新增误差条或统计显著性声明。

模板与样式：
使用已有论文的官方 neurips_2026.sty，preprint 模式，仅用于课程项目排版。
表格为 booktabs 三线表，无竖线；caption 位于表上方、图下方。
