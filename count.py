import numpy as np
import pandas as pd
from matplotlib import image
import clamutils_exp as cu

# -----------------------
# parametri come nel vecchio script
# -----------------------
k = 68
diam = 10
dt = 0.1
disc_radius = 350

threshold_dist = 1
threshold_dist_tot = 3
dist_step = 1
strict = True
only_inside = True
blinking_ratio = 0.5
lifetime_min = 0

# -----------------------
# load settings
# -----------------------
settings = np.genfromtxt("Clamidomoni_settings.txt")

pix = settings[k, 4]
ill = int(settings[k, 2])
ctr = int(settings[k, 6])
fluo = int(settings[k, 7])
lin = int(settings[k, 8])
axis = int(settings[k, 9])

# -----------------------
# build speckle
# -----------------------
if ill == 0:
    field = image.imread("/home/user/Scrivania/Videos/December2023/Quantitative/forTraj/illumination1.tiff").astype(float)
else:
    field = image.imread(f"/home/user/Scrivania/Videos/December2023/Quantitative/forTraj/illumination{ill}.tiff").astype(float)

speckle = cu.Speckle(field, diam, pix, ctrp=ctr, lin=lin, axis=axis)

filt = cu.Filters(
    lifetime_min=lifetime_min,
    threshold_dist=threshold_dist,
    threshold_dist_tot=threshold_dist_tot,
    dist_step=dist_step,
    strict=strict,
    only_inside=only_inside,
    blinking_ratio=blinking_ratio,
)

# -----------------------
# load trajectories
# -----------------------
data = np.genfromtxt("tAlgae68.txt", delimiter=",", skip_header=1)

ttot = data[:, 0].astype(int)
Ytot = data[:, 1].astype(float) * pix
Xtot = data[:, 2].astype(float) * pix
Tagstot = data[:, 3].astype(int)

particle_tags = np.sort(np.unique(Tagstot))

# -----------------------
# raw counts
# -----------------------
df_raw = pd.DataFrame({"snap": ttot, "tag": Tagstot})
raw_counts = df_raw.groupby("snap")["tag"].nunique()

# -----------------------
# counts after existance only
# -----------------------
rows_exist = []

# -----------------------
# counts after existance + cleaner
# -----------------------
rows_clean = []

for tagi in particle_tags:
    cond = Tagstot == tagi
    Xp = Xtot[cond]
    Yp = Ytot[cond]
    tp = ttot[cond]

    # pass 1: existance only
    p1 = cu.particle_class(int(tagi), tp.copy(), Xp.copy(), Yp.copy(), speckle, dt)
    p1.existance(filt)
    if p1.exist:
        for s in p1.snaps:
            rows_exist.append((s, tagi))

    # pass 2: existance + cleaner (come nello script)
    p2 = cu.particle_class(int(tagi), tp.copy(), Xp.copy(), Yp.copy(), speckle, dt)
    p2.existance(filt)
    p2.traj_cleaner(filt)
    if p2.exist:
        for s in p2.snaps:
            rows_clean.append((s, tagi))

df_exist = pd.DataFrame(rows_exist, columns=["snap", "tag"]) if rows_exist else pd.DataFrame(columns=["snap", "tag"])
df_clean = pd.DataFrame(rows_clean, columns=["snap", "tag"]) if rows_clean else pd.DataFrame(columns=["snap", "tag"])

exist_counts = df_exist.groupby("snap")["tag"].nunique() if not df_exist.empty else pd.Series(dtype=int)
clean_counts = df_clean.groupby("snap")["tag"].nunique() if not df_clean.empty else pd.Series(dtype=int)

print("RAW max count                :", raw_counts.max())
print("After existance() max count  :", exist_counts.max() if not exist_counts.empty else 0)
print("After traj_cleaner() max count:", clean_counts.max() if not clean_counts.empty else 0)

print("\nFirst raw counts:")
print(raw_counts.head(10))

print("\nFirst cleaned counts:")
print(clean_counts.head(10))