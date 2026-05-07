# moving-chlamydomonas

A tool for analysing the motion of microalgae (*Chlamydomonas*) recorded on microscopy videos.

The package provides two web interfaces:

- **`tracking_app.py`** — interactively tunes tracking parameters with a live preview, saves the configuration, and launches batch tracking on one or more videos
- **`clam-app`** — analyses the resulting trajectory files and produces statistical plots

The typical workflow is: first run **tracking** to extract trajectories from videos, then run the **analysis** on the resulting files.

---

## Requirements

- SSH access to the lab workstation
- A web browser (Chrome, Firefox, Safari, etc.)

Nothing needs to be installed on your personal computer.

---

## 1. Connecting to the workstation

All computations run on the lab workstation. To connect you need a **terminal**:

- **Mac**: open the **Terminal** app (Applications → Utilities → Terminal)
- **Linux**: open a **Terminal**
- **Windows**: open **PowerShell** (search "PowerShell" in the Start menu) or install [PuTTY](https://www.putty.org/)

In the terminal, type the following command and press **Enter**:

```
ssh YOUR_USERNAME@WORKSTATION_ADDRESS
```

> Replace `YOUR_USERNAME` with your username and `WORKSTATION_ADDRESS` with the IP address of the machine (ask the lab manager if you don't know it).

The system will ask for your password. Type it and press Enter — the characters won't appear as you type, that's normal.

When the prompt changes (it will look something like `username@machinename:~$`) you are connected.

---

## 2. Setting up the environment (first time only)

### 2.1 Navigate to the package folder

Once connected, move into the package directory:

```
cd /PATH/TO/moving_chlamydomonas
```

> Ask the lab manager for the exact path on this workstation.

### 2.2 Check whether the virtual environment already exists

Type:

```
ls .venv
```

**Case A** — you see a list of folders and files → the environment is already set up. Skip to **Section 3**.

**Case B** — you see:
```
ls: cannot access '.venv': No such file or directory
```
The environment does not exist yet. Continue with step 2.3.

### 2.3 Create the virtual environment and install the dependencies

Run the following commands **one at a time**, pressing Enter after each one and waiting for it to finish before typing the next:

```
python3 -m venv .venv
```
```
source .venv/bin/activate
```
```
pip install --upgrade pip
```
```
pip install -r requirements.txt
```
```
pip install -e .
```

Installing all packages takes a few minutes. Once you see the prompt again, it is done.

From this point on, `(.venv)` will appear at the beginning of the prompt — this means the virtual environment is active.

---

## 3. Starting a working session

**Every time** you want to use the package (after the first-time setup), do the following:

**1.** Open a terminal and connect to the workstation (as in Section 1).

**2.** Activate the virtual environment:

```
cd /PATH/TO/moving_chlamydomonas
source .venv/bin/activate
```

`(.venv)` will appear at the beginning of the prompt. You are now ready to launch the tracking or the analysis app.

---

## 4. Tracking: tuning and batch launch (`tracking_app.py`)

The tracking app lets you visually tune the detection parameters on a single frame before running the full batch. It also saves the configuration and launches the tracking jobs — all from the browser.

### 4.1 Opening the SSH tunnel and launching the app

Open a **new** terminal window and connect to the workstation using this command (note the `-L` flag, which creates the tunnel for the web interface):

```
ssh -L 8502:localhost:8502 YOUR_USERNAME@WORKSTATION_ADDRESS
```

Once connected, move into the `tracking_from_video` subfolder and activate the virtual environment:

```
cd /PATH/TO/moving_chlamydomonas/tracking_from_video
source ../.venv/bin/activate
```

Then start the app:

```
streamlit run tracking_app.py --server.port 8502
```

You should see:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8502
```

Open your browser and go to:

```
http://localhost:8502
```

### 4.2 Using the interface

The app is divided into numbered sections:

**1. Video** — enter the full path to one of your `.avi` video files and select the colour channel (default: `2`). Example:
```
/home/YOUR_USERNAME/nas_public/tAlgae_exp/tAlgae68.avi
```

**2. Background** — choose whether to subtract the background before tracking (recommended: enabled, method: `mean`).

**3. Tuning `tp.locate`** — adjust the three main detection parameters:

| Parameter | Meaning | What to do if results are poor |
|---|---|---|
| `diameter` | Estimated size of the algae in pixels (**must be odd**) | Increase if algae appear larger than the detected circles; decrease if circles are too big |
| `minmass` | Minimum brightness threshold to detect a particle | Increase to remove false detections; decrease if algae are being missed |
| `invert` | Invert the image | Enable only if algae are dark on a bright background |

**4. Preview** — use the frame slider to inspect any frame of the video. The left panel shows the detected particles overlaid on the image. The right panel lets you compare several `minmass` values side by side — click **Genera griglia confronto** to generate the comparison grid.

**5. Link parameters** — controls how detected particles are connected between frames:

| Parameter | Meaning | Typical value |
|---|---|---|
| `search_range` | Maximum displacement allowed between consecutive frames (pixels) | `15` |
| `adaptive_stop` | Minimum search range before giving up on linking | `5` |
| `adaptive_step` | Reduction factor for the adaptive search | `0.98` |
| `filter_stubs` | Discard trajectories shorter than this number of frames | `0` |

**6. Save config** — when the parameters look good, give the configuration a name (e.g. `config_exp68`) and click **Salva**. The file is saved in `tracking_from_video/configs/`. You can save multiple configs with different parameters and assign them to different videos in the next step.

**7. Batch tracking launch** — enter the folder containing the videos and the output folder, then assign each video to a configuration using the expandable panels. Click **▶ Avvia tutti i gruppi** to start.

The video list is automatically filtered by the machine selected at the top of the page:

| Machine | Video naming convention | Example |
|---|---|---|
| `microscope2D` | `tAlgae*.avi` | `tAlgae68.avi` |
| `photonicsLab` | `tPhot*.avi` | `tPhot12.avi` |
| `lightfield` | `td*.avi` | `td5.avi` |

Example paths:

| Field | Example |
|---|---|
| Cartella video | `/home/YOUR_USERNAME/nas_public/tAlgae_exp/` |
| Cartella output tracking | `/home/YOUR_USERNAME/tracking_results/` |

When tracking is complete, you will find one `.txt` file and one `.h5` file for each processed video in the output folder.

---

## 5. Registering new experiments (`Clamidomoni_exp_setter.py`)

Before running the analysis, every experiment must be registered in a central settings file called `Clamidomoni_settings.txt`. This file is **generated by running a Python script** — never edited by hand.

> **Where is this script?** It lives outside the package, in a shared location on the workstation. Ask the lab manager for the exact path.

### 5.1 How it works

The script is **cumulative**: it records every experiment the lab has ever done, from ID 0 up to `n_exps`. Every time you acquire a new batch of videos, you open the script, append a few lines for the new experiment IDs, update `n_exps`, and re-run it. The script regenerates `Clamidomoni_settings.txt` from scratch each time.

The resulting table has one row per experiment ID and the following columns:

| Column | Meaning | Default |
|---|---|---|
| `exps` | Experiment number (filled automatically) | — |
| `exp_type` | Type of experiment: `0`=none, `1`=normal, `2`=gaussian, `3`=outside ring, `4`=single cell, `5`=LED | `0` |
| `ill` | Illumination image number (matches the `illumination*.tiff` filename) | `0` |
| `int` | Illumination intensity (µW) | `0` |
| `pix_arr` | Pixel size (µm/pixel) | `0` |
| `no_skip` | Include in analysis: `1`=yes (default), `0`=skip | `1` |
| `ctr` | Illumination centre position: `0`=centre, `1`=right, `-1`=left | `0` |
| `fluo` | Fluorescence image: `0`=none, `1`=ring, `2`=gaussian | `0` |
| `lin` | Linear illumination (gradient): `0`=no, `1`=yes | `0` |
| `axis` | Axis of the linear gradient: `0`=x, `1`=y | `0` |

You only need to set the columns that differ from the default. Everything else is automatically zero (or one for `no_skip`).

### 5.2 Structure of the script

The script is organised in fixed blocks. The **header** is always the same — only `n_exps` changes:

```python
import numpy as np
import matplotlib.pyplot as plt
import os.path
from tqdm import tqdm
from celluloid import Camera
from matplotlib import image
from scipy.spatial.distance import cdist
from math import dist
import matplotlib as mpl

n_exps = 465   # ← total number of experiments (update this when you add new ones)

exps    = np.arange(n_exps+1)
ill     = np.zeros((n_exps+1))   # illumination name
int     = np.zeros((n_exps+1))   # illumination intensity in µW
pix_arr = np.zeros((n_exps+1))   # µm/pixel ratio
no_skip = np.ones((n_exps+1))    # 1=include, 0=skip
exp_type= np.zeros((n_exps+1))   # experiment type
ctr     = np.zeros((n_exps+1))   # illumination centre offset
fluo    = np.zeros((n_exps+1))   # fluorescence image type
lin     = np.zeros((n_exps+1))   # linear illumination
axis    = np.zeros((n_exps+1))   # gradient axis
```

After the header come the assignment blocks (one per column), and at the very end the two lines that produce the file — these never change:

```python
settings = np.vstack([exps, exp_type, ill, int, pix_arr, no_skip, ctr, fluo, lin, axis]).transpose()
np.savetxt("Clamidomoni_settings.txt", settings, fmt=["%d","%d","%d","%d","%.6f","%d","%d","%d","%d","%d"])
```

### 5.3 Adding a new batch of experiments

Suppose you acquired experiments 466 to 472 on 5 May, then 473 to 480 on 6 May — all with the same microscope settings. Here is what you do:

**1. Update `n_exps`** at the top of the script. `n_exps` is the **index of the last experiment**, not the total count — if the last video you acquired is number 480, then `n_exps = 480`:
```python
n_exps = 480   # was 465
```

**2. Append one line to each block** for the new range. The critical rule is:

> **The end index in `[start:end]` is excluded.**
> `ill[466:481] = 7` sets experiments 466, 467, …, 480. Experiment 481 is NOT included.
> To cover experiments 466–480, always write `481` on the right.

Add to the `ill` block:
```python
ill[466:481] = 7   # illumination 7 for experiments 466–480
```

Add to the `int` block:
```python
int[466:481] = 180
```

Add to the `pix_arr` block:
```python
pix_arr[466:481] = 1/0.96
```

Add to the `exp_type` block:
```python
exp_type[466:481] = 1   # normal experiment
```

**3. Skip bad videos** (if any) by setting `no_skip=0` for individual IDs:
```python
no_skip[471] = 0   # video 471 was bad, exclude it
```

**4. Set `ctr`, `fluo`, `lin`, `axis`** only if they differ from zero:
```python
ctr[466:481] = 1   # illumination centre shifted to the right
```
If they are the same as the default (zero), you do not need to add anything.

### 5.4 Running the script

Navigate to the folder where the script is located, activate the virtual environment, and run it:

```
source /PATH/TO/moving_chlamydomonas/.venv/bin/activate
cd /PATH/TO/script/folder
python Clamidomoni_exp_setter.py
```

The script overwrites `Clamidomoni_settings.txt` in the same folder. **This is the file you will point to in the analysis app** (field: *File settings*).

### 5.5 Common mistakes

**Off-by-one error** — the most frequent mistake. If your experiments go from 466 to 480 (inclusive), the slice must be `[466:481]`, not `[466:480]`. A quick check: `481 − 466 = 15`, which is the correct number of experiments (466, 467, …, 480).

**Forgetting to update `n_exps`** — if a new experiment ID exceeds the current `n_exps`, the array is too short. The assignment will either be silently ignored or raise an error. Always update `n_exps` first.

**Running the script from the wrong folder** — `Clamidomoni_settings.txt` is saved in whatever folder you are in when you run the script. Always `cd` into the script's folder first.

---

## 6. Analysis: the web interface (`clam-app`)

The analysis runs as a separate web page, also on the workstation. It uses port `8501`, so it can run at the same time as the tracking app without conflicts.

### 6.1 Opening the SSH tunnel and launching the app

Open another **new** terminal window and connect with:

```
ssh -L 8501:localhost:8501 YOUR_USERNAME@WORKSTATION_ADDRESS
```

Activate the virtual environment:

```
cd /PATH/TO/moving_chlamydomonas
source .venv/bin/activate
```

Start the app:

```
clam-app
```

Open your browser and go to:

```
http://localhost:8501
```

> If the page does not load, make sure `clam-app` is still running in the terminal (the prompt should not have returned).

### 6.2 Filling in the fields

**Section "Macchina" (Machine)**
Select which instrument was used to acquire the data: `superK` or `laserGlow`.

**Section "Percorsi" (Paths)**

| Field | What to enter |
|---|---|
| Cartella traiettorie | Path to the folder where the tracking saved the `.txt` files — e.g. `/home/YOUR_USERNAME/tracking_results/` |
| Cartella illuminazione | Path to the folder containing the `illumination*.tiff` files — e.g. `/home/YOUR_USERNAME/nas_public/illumination/` |
| File settings | Path to the `Clamidomoni_settings.txt` file — e.g. `/home/YOUR_USERNAME/nas_public/Clamidomoni_settings.txt` |
| Cartella mappa radiale | Path to the radial map (leave empty if not used) |
| Cartella output risultati | Where to save the analysis results — e.g. `/home/YOUR_USERNAME/results/` |
| Experiment ID | The experiment number (e.g. `68`). To analyse multiple experiments together and also obtain the merged output, separate the numbers with a comma: `68, 69, 73` |

**Analysis parameters** (expandable sections)

The default values are the ones normally used — do not change them unless you know what you are doing.

### 6.3 Running the analysis

Click the **▶ Avvia analisi** button.

Processing may take a few minutes depending on the number of trajectories. When it is done, a green message will appear with the path where results have been saved, followed by plots in the following tabs:

- **Theta** — orientation angle distribution
- **Flux** — particle flux
- **N(t)** — number of particles over time
- **Angle** — angle distribution
- **Omega** — angular velocity distribution
- **Density vs dist** — density as a function of distance from the centre
- **Tumbling θ** — tumbling events as a function of the light gradient
- **Tumbling duration** — tumbling duration as a function of distance

---

## 7. Closing everything

When you are done:

1. In each terminal where an app is running, press `Ctrl+C` to stop it.
2. Type `deactivate` to deactivate the virtual environment.
3. Type `exit` to close the SSH connection.

---

## Repository structure

```
moving_chlamydomonas/
├── app.py                              # Analysis web interface (clam-app)
├── clam_pipeline/                      # Core analysis module
│   ├── analyze.py
│   ├── cli.py
│   ├── io.py
│   └── plots.py
├── analyses/                           # Per-plot functions
├── tracking_from_video/
│   ├── tracking_app.py                 # Tracking web interface
│   ├── configs/
│   │   └── config_tracking.json       # Default tracking config (copy and rename for each experiment)
│   └── src/
│       ├── tracking_algae.py          # Batch tracking script (used internally by the app)
│       └── tracking_core.py          # Internal tracking functions
├── pyproject.toml
└── requirements.txt
```

---

## Troubleshooting

**"command not found: clam-app"**
You forgot to activate the virtual environment. Run:
```
source .venv/bin/activate
```

**The page http://localhost:8501 or http://localhost:8502 does not open**
Check that the app is still running in the terminal (the prompt should not have returned). Also verify that the SSH command included the `-L ...` flag with the correct port number.

**"Impossibile importare clam_pipeline" error in the analysis interface**
The app was started from the wrong folder, or the package was not installed with `pip install -e .`. Go back to Section 2.

**Tracking detects too many particles (false positives)**
Increase `minmass` in the tracking app. Use the comparison grid (Section 4.2, step 4) to find a good value visually.

**Tracking detects too few particles (algae being missed)**
Decrease `minmass`, or check that `diameter` matches the actual size of the algae in the video.
