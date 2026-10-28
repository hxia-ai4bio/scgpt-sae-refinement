论文 v8 — Interpreting and Refining scGPT Cell Embeddings with Structure-Preserving Sparse Autoencoders
Authors: Xia Haoxiang, Chang Runjia, Xiao Jiahao

Overleaf：New Project → Upload Project → 上传 ZIP；主文档选择 main.tex。
编译器可使用 pdfLaTeX，参考文献由 BibTeX 自动生成。首次编译需完成交叉引用更新。
使用 NeurIPS 2026 官方样式的 preprint 模式；本文件为 BS6204 课程项目，并不表示会议投稿或录用。

内容：六节正文、六张图、一张可编辑三线表、16 篇参考文献。
1 Introduction
2 Related Work
3 Methodology
4 Experiments
5 Discussion
6 Conclusion and Future Work
Code Availability（不编号）
References（自动生成，不编号）

main.tex：正文、作者、页眉页脚设置。
references.bib：16 篇文献，正文采用作者—年份自动引用。
neurips_2026.sty：会议样式文件。
images/：Figure 1–6；Figure 5–6 为矢量 PDF。
tables/classification.tex：Table 1，可编辑 LaTeX 三线表。

v7 在 v6 基础上补充 Section 4.1：数据来源标识、实际纳入细胞数、供体/批次划分、SAE 维度、Ridge 分类及 Leiden 聚类设置。PBMC 使用实际输入文件和批次标识；其完整下载出处仍需追溯。
首页不设页眉，后续页面使用简短标题；全篇页码居中，移除旧 Working draft 标记。
Table 1 的 best-observed 选择说明保留。
正式期刊论文、会议论文、技术报告和预印本在参考文献中按实际来源标记。

GitHub: https://github.com/hxia-ai4bio/scgpt-sae-refinement
仓库已公开发布项目代码、结果汇总、绘图脚本和运行说明。原始数据和模型权重需另行配置。

v8 在 Conclusion and Future Work 之后、References 之前增加英文 Code Availability，链接至公开 GitHub 仓库。
