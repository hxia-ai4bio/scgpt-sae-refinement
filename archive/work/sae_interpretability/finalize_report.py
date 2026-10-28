from pathlib import Path
p=Path('work/sae_interpretability/10_report.py')
s=p.read_text()
s=s.replace("      cards.append(f'<section>","      assoc_label='跨类型表达关联（加性控制细胞类型与供体）' if f==1724 else '同类型内表达关联'\n      cards.append(f'<section>")
s=s.replace('<b>同类型内表达关联：</b>','<b>{assoc_label}：</b>')
s=s.replace("      '\\n\\n'.join(textcards)+'\\n\\n'","      + '\\n\\n'.join(textcards)+'\\n\\n'")
s=s.replace('SAE20轮尚未充分收敛','SAE采用20轮预算，未验证充分收敛')
s=s.replace("      '通路：采用Hallmark", "      '通路：训练集表达关联基因的富集在匹配表达量与检出率的随机对照下，没有条目达到FDR<0.05（最小q约0.0703）；归因基因的显著条目主要集中于相互重叠的翻译/核糖体相关通路。采用Hallmark")
p.write_text(s)
