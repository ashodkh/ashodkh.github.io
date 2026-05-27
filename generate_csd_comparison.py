import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import csd
import torch
from torch import optim
from nflows import transforms, distributions, flows
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

DATA_DIR = '/Users/ASK126/Desktop/research/gw_challenge/Datasets'
OUT_PATH = '/Users/ASK126/Desktop/research/git_repos/ashodkh.github.io/images/csd_comparison.png'

print("Loading data...")
background = np.load(f'{DATA_DIR}/background.npz')['data']
bbh = np.load(f'{DATA_DIR}/bbh_for_challenge.npy')

sglf = np.load(f'{DATA_DIR}/sglf_for_challenge.npy')

N_total = background.shape[0]
print("Computing CSDs...")
background_cs = np.zeros((N_total, 101))
bbh_cs = np.zeros((bbh.shape[0], 101))
sglf_cs = np.zeros((sglf.shape[0], 101))
for i in range(N_total):
    f, Pxx = csd(background[i, 0, :], background[i, 1, :])
    background_cs[i] = np.abs(Pxx)
    f, Pxx = csd(bbh[i, 0, :], bbh[i, 1, :])
    bbh_cs[i] = np.abs(Pxx)
    f, Pxx = csd(sglf[i, 0, :], sglf[i, 1, :])
    sglf_cs[i] = np.abs(Pxx)

print("Preprocessing...")
scaler = StandardScaler()
x_scaled = scaler.fit_transform(background_cs)
x_train, x_val = train_test_split(x_scaled, test_size=0.2, random_state=42)

pca = PCA(n_components=70)
x_train_pca = pca.fit_transform(x_train)
x_val_pca = pca.transform(x_val)

print("Building flow...")
d = x_train_pca.shape[1]
base_dist = distributions.StandardNormal(shape=[d])
transforms_array = []
for _ in range(3):
    transforms_array.append(transforms.ReversePermutation(features=d))
    transforms_array.append(transforms.MaskedAffineAutoregressiveTransform(features=d, hidden_features=16))
transform = transforms.CompositeTransform(transforms_array)
flow = flows.Flow(transform, base_dist)

optimizer = optim.Adam(flow.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 800], gamma=0.1)

print("Training flow...")
batch_size = 5000
n_epochs = 60
N_train = x_train_pca.shape[0]
n_batches = N_train // batch_size
for epoch in range(n_epochs):
    flow.train()
    for _ in range(n_batches):
        batch_inds = np.random.choice(N_train, size=batch_size, replace=False)
        x_batch = torch.tensor(x_train_pca[batch_inds], dtype=torch.float32)
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=x_batch).mean()
        loss.backward()
        optimizer.step()
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        flow.eval()
        val_loss = -flow.log_prob(torch.tensor(x_val_pca, dtype=torch.float32)).mean().item()
        print(f"  Epoch {epoch+1}/{n_epochs}, Val Loss: {val_loss:.4f}")

print("Computing log-probabilities and latents...")
flow.eval()
with torch.no_grad():
    x_bg_tensor = torch.tensor(pca.transform(x_scaled), dtype=torch.float32)
    background_log_probs = flow.log_prob(x_bg_tensor).numpy()
    background_latents = flow.transform_to_noise(x_bg_tensor).numpy()
    bbh_log_probs = flow.log_prob(
        torch.tensor(pca.transform(scaler.transform(bbh_cs)), dtype=torch.float32)
    ).numpy()
    sglf_log_probs = flow.log_prob(
        torch.tensor(pca.transform(scaler.transform(sglf_cs)), dtype=torch.float32)
    ).numpy()

# Pick representative samples by log-probability
i_noise = np.argmax(background_log_probs)   # most typical noise (highest log-prob)
i_bbh   = np.argmin(bbh_log_probs)          # clearest BBH (lowest log-prob)

print(f"Noise sample index {i_noise}: log_prob = {background_log_probs[i_noise]:.2f}")
print(f"BBH sample index  {i_bbh}:   log_prob = {bbh_log_probs[i_bbh]:.2f}")

# --- ROC curves via log-probability threshold ---
# Lower log-prob = more anomalous = more likely a GW signal.
# Use sklearn's roc_curve so the vertical segment at FPR=0 is handled correctly.
from sklearn.metrics import roc_curve, roc_auc_score

N_bg = len(background_log_probs)
# Negate so that higher score = more likely GW (sklearn convention)
scores_bbh  = -np.concatenate([background_log_probs, bbh_log_probs])
scores_sglf = -np.concatenate([background_log_probs, sglf_log_probs])
labels_bbh  = np.concatenate([np.zeros(N_bg), np.ones(len(bbh_log_probs))])
labels_sglf = np.concatenate([np.zeros(N_bg), np.ones(len(sglf_log_probs))])

fpr_bbh,  tpr_bbh,  _ = roc_curve(labels_bbh,  scores_bbh)
fpr_sglf, tpr_sglf, _ = roc_curve(labels_sglf, scores_sglf)
auc_bbh  = roc_auc_score(labels_bbh,  scores_bbh)
auc_sglf = roc_auc_score(labels_sglf, scores_sglf)
print(f"AUC BBH:  {auc_bbh:.4f}")
print(f"AUC SGLF: {auc_sglf:.4f}")

# Frequencies from scipy.signal.csd (same as used during CSD computation)
f_dummy, _ = csd(background[0, 0, :], background[0, 1, :])

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#1a1a2e')

colors = ['#8dd3c7', '#fa8174']

ax0, ax1 = axes

ax0.plot(f_dummy, background_cs[i_noise], color=colors[0], lw=2)
ax0.set_title('Background Noise', color='white', fontsize=14)
ax0.set_xlabel('Frequency (normalized)', color='white', fontsize=12)
ax0.set_ylabel('|CSD|', color='white', fontsize=12)
ax0.tick_params(colors='white')
for spine in ax0.spines.values():
    spine.set_edgecolor('#444466')

ax1.plot(f_dummy, bbh_cs[i_bbh], color=colors[1], lw=2)
ax1.set_title('BBH Signal', color='white', fontsize=14)
ax1.set_xlabel('Frequency (normalized)', color='white', fontsize=12)
ax1.set_ylabel('|CSD|', color='white', fontsize=12)
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#444466')

fig.suptitle('Cross-Spectral Density: Noise vs BBH Signal', color='white', fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved to {OUT_PATH}")

# --- Latent space histograms + correlation matrix ---
import matplotlib.gridspec as gridspec

LATENT_OUT = '/Users/ASK126/Desktop/research/git_repos/ashodkh.github.io/images/latent_histograms.png'
dims = [0, 5, 20, 50]
x_gauss = np.linspace(-5, 5, 500)
gauss_pdf = np.exp(-0.5 * x_gauss ** 2) / np.sqrt(2 * np.pi)
hist_color = '#8dd3c7'
gauss_color = '#fdb462'

# Correlation matrix of latent variables; mask diagonal so colorscale reflects off-diagonal range
corr_matrix = np.corrcoef(background_latents.T)  # shape (70, 70)
corr_plot = corr_matrix.copy()
np.fill_diagonal(corr_plot, np.nan)
off_max = np.nanmax(np.abs(corr_plot))
print(f"Off-diagonal max |corr|: {off_max:.4f}, mean: {np.nanmean(np.abs(corr_plot)):.4f}")

fig2 = plt.figure(figsize=(16, 12), facecolor='#1a1a2e')
gs = gridspec.GridSpec(2, 4, figure=fig2, height_ratios=[1, 1.5], hspace=0.45, wspace=0.35)

# Top row: histograms
for col, dim in enumerate(dims):
    ax = fig2.add_subplot(gs[0, col])
    ax.set_facecolor('#1a1a2e')
    ax.hist(background_latents[:, dim], bins=80, density=True,
            histtype='step', color=hist_color, lw=2, label='Latent')
    ax.plot(x_gauss, gauss_pdf, color=gauss_color, lw=2, label='$\\mathcal{N}(0,1)$')
    ax.set_title(f'Dimension {dim}', color='white', fontsize=13)
    ax.set_xlabel('$z$', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.set_xlim(-5, 5)
    if col == 0:
        ax.set_ylabel('Density', color='white', fontsize=12)
        ax.legend(fontsize=11, framealpha=0.3,
                  labelcolor='white', facecolor='#1a1a2e', edgecolor='#444466')

# Bottom row: correlation matrix spanning middle two columns
ax_corr = fig2.add_subplot(gs[1, 1:3])
ax_corr.set_facecolor('#1a1a2e')
im = ax_corr.imshow(corr_plot, cmap='RdBu_r', vmin=-off_max, vmax=off_max, aspect='auto')
cbar = fig2.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color='white')
cbar.outline.set_edgecolor('#444466')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
ax_corr.set_title('Latent Cross-Correlation Matrix', color='white', fontsize=13)
ax_corr.set_xlabel('Latent dimension', color='white', fontsize=12)
ax_corr.set_ylabel('Latent dimension', color='white', fontsize=12)
ax_corr.tick_params(colors='white')
for spine in ax_corr.spines.values():
    spine.set_edgecolor('#444466')

fig2.suptitle('Latent Space Diagnostics', color='white', fontsize=15, y=1.01)
fig2.savefig(LATENT_OUT, dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
print(f"Saved to {LATENT_OUT}")

# --- ROC curve figure ---
ROC_OUT = '/Users/ASK126/Desktop/research/git_repos/ashodkh.github.io/images/ROC_curve.png'

fig3, ax3 = plt.subplots(figsize=(7, 7))
fig3.patch.set_facecolor('#1a1a2e')
ax3.set_facecolor('#1a1a2e')

ax3.plot(fpr_bbh,  tpr_bbh,  color='#fa8174', lw=2.5, label=f'BBH  (AUC = {auc_bbh:.2f})')
ax3.plot(fpr_sglf, tpr_sglf, color='#8dd3c7', lw=2.5, label=f'SGLF (AUC = {auc_sglf:.2f})')
ax3.plot([0, 1], [0, 1], color='#666688', lw=1.5, linestyle='--', label='Random')

ax3.set_xlabel('False Positive Rate', color='white', fontsize=13)
ax3.set_ylabel('True Positive Rate', color='white', fontsize=13)
ax3.set_title('ROC Curves', color='white', fontsize=14)
ax3.tick_params(colors='white')
ax3.legend(fontsize=12, framealpha=0.3, labelcolor='white',
           facecolor='#1a1a2e', edgecolor='#444466')
for spine in ax3.spines.values():
    spine.set_edgecolor('#444466')
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)

fig3.tight_layout()
fig3.savefig(ROC_OUT, dpi=150, bbox_inches='tight', facecolor=fig3.get_facecolor())
print(f"Saved to {ROC_OUT}")
