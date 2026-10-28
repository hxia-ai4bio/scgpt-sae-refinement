"""Redraw the approved schematic with real typography and vector primitives.
All cells and bars are illustrative, not experimental results.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import findfont

OUT = Path("/Users/apple/Documents/Codex/2026-10-18/wo/outputs/sae_figures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "Arial", "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white",
})
print("Font:", findfont("Arial", fallback_to_default=False))
W, H = 1440, 1220
fig = plt.figure(figsize=(12, 12 * H / W), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
ax.set_aspect("equal")
ax.axis("off")
NAVY, TEXT, BLUE = "#20344A", "#38434F", "#229BE0"
LAV, GREEN = "#AA85C6", "#4DA678"

def text(x, y, s, size=26, bold=False, color=TEXT, ha="center", **kw):
    return ax.text(x, y, s, fontsize=size * .6, weight="bold" if bold else "normal",
                   color=color, ha=ha, va="center", **kw)

def line(x1, y1, x2, y2, color=NAVY, lw=1.6, **kw):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="round", **kw)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=15, lw=1.6, color=NAVY,
                                shrinkA=0, shrinkB=0))

def rounded(x, y, w, h, fill, stroke="none", radius=14, lw=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=fill, edgecolor=stroke, linewidth=lw))

def cell(x, y, r=35, kind="blue"):
    if kind == "blue":
        ax.add_patch(Circle((x, y), r, fc="#DDF2FB", ec=BLUE, lw=1.5, zorder=4))
        ax.add_patch(Circle((x, y), r * .39, fc=BLUE, ec="none", zorder=5))
    elif kind == "A":
        ax.add_patch(Circle((x, y), r, fc="#EAE1F3", ec=LAV, lw=1.3, zorder=4))
        ax.add_patch(Circle((x, y), r * .48, fc="#9496B3", ec="none", zorder=5))
    elif kind == "B":
        ax.add_patch(Ellipse((x, y), 2*r, 1.65*r, fc="#EEDFF5", ec=LAV, lw=1.3, zorder=4))
        ax.add_patch(Circle((x-3, y+1), r*.39, fc="#AC7FC6", ec="none", zorder=5))
    else:
        th = np.linspace(0, 2*np.pi, 180)
        radius = r * (1 + (.065*np.cos(10*th) if kind == "C" else .12*np.cos(3*th+.5)))
        yy = radius*np.sin(th)*(1 if kind == "C" else .86)
        ax.add_patch(Polygon(np.c_[x+radius*np.cos(th), y+yy], closed=True,
                             fc="#E1F0E2" if kind=="C" else "#E0E7EE",
                             ec=GREEN if kind=="C" else "#8A9BB7", lw=1.3, zorder=4))
        if kind == "C":
            ax.add_patch(Ellipse((x-2,y), 25, 33, angle=-35, fc=GREEN, ec="none", zorder=5))
            ax.add_patch(Circle((x+9,y-9), 10, fc="#E1F0E2", ec="none", zorder=6))
        else:
            ax.add_patch(Ellipse((x+3,y-2), 25, 26, fc="#8A9BB7", ec="none", zorder=5))

def vector(x, top, colors, edges=None, step=33, r=13):
    height = (len(colors)-1)*step + 2*r + 16
    ax.add_patch(Polygon([(x-27,top),(x+27,top),(x+27,top+height),(x-27,top+height)],
                         fc="white", ec=NAVY, lw=1.5))
    for n, color in enumerate(colors):
        edge = edges[n] if edges else NAVY
        ax.add_patch(Circle((x,top+8+r+n*step),r,fc=color,ec=edge,lw=1))
    return top+height

text(W/2, 54, "Sparse Reconstruction for Embedding Refinement", 40, True, NAVY)

# Left: frozen upstream embedding followed by sparse encoding and reconstruction.
cell(105, 265, 35)
text(105, 325, r"Cell $i$", 27, True, NAVY)
arrow(153,265,198,265)
ax.add_patch(Polygon([(205,210),(337,237),(337,293),(205,320)], closed=True,
                     fc="#ECE8F4", ec="#8E83AA", lw=1.6))
text(267,255,"scGPT",27,True,NAVY)
text(267,284,"Encoder",23)
arrow(345,265,390,265)
base = ["#CDEFFC","#97A4BB","#B487CD","#BDE6F7","#92A1B7","#ECE0F5"]
edges = [BLUE,"#92A1B7",LAV,BLUE,"#92A1B7",LAV]
bottom_h = vector(430,159,base,edges)
text(430,bottom_h+33,r"$h_i$",40)
bottom_hat = vector(430,624,[base[0],base[1],base[2],base[3],base[5],base[4]],edges)
text(478, bottom_hat-28,r"$\hat{h}_i$",40,ha="left")
z_top=462
bottom_z=vector(160,z_top,["white","#E9AB33","white","white","white","#E9AB33","white","white"],step=28,r=12)
text(160,423,"Sparse\nFeatures",26,True,NAVY,linespacing=1.15)
text(160,bottom_z+30,r"$z_i$",40)
arrow(401,365,203,500)
text(290,414,"SAE Encoder",24,rotation=34)
arrow(202,650,385,743)
text(291,675,"SAE Decoder",24,rotation=-27)

# MSE: a bracket between original and reconstructed vectors.
line(471,356,497,356)
line(497,356,497,466)
line(497,570,497,639)
line(471,639,497,639)
rounded(392,466,210,104,"#FCF1E5")
text(497,493,"MSE",28,True,NAVY)
text(497,538,r"$\Vert h_i-\hat{h}_i\Vert_2^2$",32)

# Compact two-row cell neighborhoods; selected cells share exactly one x coordinate.
def neighborhood(y, reconstructed=False):
    ax.add_patch(Ellipse((868,y+10),350,292,fc="#EAF7F1" if reconstructed else "#EBF6FB",ec="none",zorder=0))
    points=[(856,y-75),(765,y+104),(958,y-8),(946,y+99)]
    weights=[4.0,3.3,.9,1.05] if not reconstructed else [3.5,3.7,1.0,1.3]
    edgecolors=["#96A4C6", "#B79ACC", "#B4BED6", "#B4BED6"] if not reconstructed else ["#79B6D1","#B79ACC","#A6C9C5","#A6C9C5"]
    for (xx, yy),lw,col in zip(points,weights,edgecolors):
        # Segment is clipped to the approximate membrane radius at each endpoint.
        v=np.array([xx-765,yy-y]); length=np.linalg.norm(v); u=v/length
        start=np.array([765,y])+u*35
        end=np.array([xx,yy])-u*31
        line(*start,*end,color=col,lw=lw,zorder=2)
    cell(765,y,35)
    text(711,y,r"$i$",38)
    for name,(xx,yy) in zip("ABCD",points):
        cell(xx,yy,30 if name=="A" else 33,name)
        text(xx+43,yy+(-10 if name=="A" else 11),name,27,color=NAVY,ha="left")
    arrow(473,y,675,y)
    arrow(1050,y,1111,y)

text(871,135,"Original Cell–Cell Similarity",27,True,NAVY)
text(871,600,"Reconstructed Cell–Cell Similarity",27,True,NAVY)
neighborhood(265)
neighborhood(730,True)
text(866,503,r"$\mathcal{L}=\mathcal{L}_{\mathrm{MSE}}+\lambda\mathcal{L}_{\mathrm{KL}}$",37,color=NAVY)

def probability_chart(y, q=False):
    x=1138; widths=39; gap=16; base_y=y+60
    heights=[91,53,30,15] if not q else [86,70,26,41]
    for j,(lab,h) in enumerate(zip("ABCD",heights)):
        bx=x+j*(widths+gap)
        ax.add_patch(Polygon([(bx,base_y),(bx+widths,base_y),(bx+widths,base_y-h),(bx,base_y-h)],
                             fc="#80BEA0" if q else "#6EBCE7",ec="#4E9B77" if q else "#3898D1",lw=1.2))
        text(bx+widths/2,base_y+26,lab,25)
    line(x-10,base_y,x+4*(widths+gap)-gap+10,base_y,lw=1.3)
    text(1240,y-94,r"$Q_i$" if q else r"$P_i$",43,color=NAVY)

probability_chart(265)
probability_chart(730,True)
line(1240,375,1240,455)
line(1240,565,1240,601)
line(1233,375,1247,375)
line(1233,601,1247,601)
rounded(1124,465,232,105,"#FCF1E5")
text(1240,493,"Similarity KL",27,True,NAVY)
text(1240,537,r"$D_{\mathrm{KL}}(P_i\Vert Q_i)$",31)

# Evaluation: generous separation from neighborhoods, 2 x 2 compact icons.
rounded(295,958,1030,222,"#FCFAF7","#E3D7C9",18,1.1)
arrow(430,bottom_hat+4,430,957)
text(327,991,"Downstream Evaluation",29,True,NAVY,ha="left")

def dots(points,color,r=6):
    for x,y in points:
        ax.add_patch(Circle((x,y),r,fc=color,ec="none"))

# Classification mini-scene.
dots([(370,1068),(384,1053),(399,1067),(380,1085),(406,1085)],BLUE)
dots([(454,1067),(469,1055),(484,1071),(451,1086),(476,1088)],LAV)
line(438,1041,420,1100,lw=1.2,linestyle="--")
text(524,1070,"Classification",25,ha="left")
# Clustering mini-scene.
for (cx,cy),col in [((846,1077),"#6AC5DF"),((883,1056),"#72B999"),((934,1077),"#AE8CCB")]:
    dots([(cx-8,cy),(cx+5,cy-8),(cx+10,cy+9),(cx-3,cy+15)],col)
text(978,1070,"Clustering",25,ha="left")
# Robustness two short strips.
for row, yy in enumerate([1123,1152]):
    rounded(355,yy-11,146,23,"white","#7C8DA2",4,.9)
    for j,c in enumerate([base[0],base[1],base[2],base[5],base[4],base[3]]):
        cx=369+j*24
        patch=Circle((cx,yy),8,fc="white" if row and j<2 else c,
                     ec="#D89B2C" if row and j<2 else "#8F9FB4",lw=.8,
                     linestyle="--" if row and j<2 else "-")
        ax.add_patch(patch)
text(524,1137,"Robustness",25,ha="left")
# Expression prediction: paired schematic bars.
for j,(a,b) in enumerate([(27,39),(20,30),(32,43)]):
    xx=836+j*40
    ax.add_patch(Polygon([(xx,1159),(xx+13,1159),(xx+13,1159-a),(xx,1159-a)],
                         fc="#E9B6A7",ec="#BF826F",lw=.8))
    ax.add_patch(Polygon([(xx+17,1159),(xx+30,1159),(xx+30,1159-b),(xx+17,1159-b)],
                         fc="none",ec="#8797AA",lw=.8,ls="--"))
line(829,1159,959,1159,lw=1.1)
text(978,1137,"Gene Expression\nPrediction",25,ha="left",linespacing=1.18)

# Save without automatic cropping: preserve exact centered title and canvas.
stem=OUT/"Figure1B_vector_refinement_v12"
fig.savefig(stem.with_suffix(".svg"))
fig.savefig(stem.with_suffix(".pdf"))
fig.savefig(stem.with_suffix(".png"),dpi=300)
fig.savefig("/Users/apple/Documents/Codex/2026-10-18/wo/work/sae_figures/v12_preview.png",dpi=120)
print("Saved:",stem)
