import matplotlib.pyplot as plt
# from matplotlib.pyplot import rcParams

# colorblind friendly palette
from cycler import cycler
# cp = ["#172A5A", "#FF7171", "#227567", "#34BAEA", "#F9D466"]
cp = ['#8C1515', '#9DBAD8', '#3667C6', '#585754']
plt.rcParams['axes.prop_cycle'] = cycler(color=cp)

# set default line width
plt.rcParams['lines.linewidth'] = 1

# set default font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Open Sans', 'Arial']
#plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# automatically despine
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# set defualt DPI
plt.rcParams['figure.dpi'] = 72


def pval_printer(pval):
    if pval >= 0.001:
        return f'p = {pval:0.3f}'
    return 'p < 0.001'


def pval_stars(pval):
    if pval < 0.001:
        return '***'
    if pval < 0.01:
        return '**'
    if pval < 0.05:
        return '*'
    return ''