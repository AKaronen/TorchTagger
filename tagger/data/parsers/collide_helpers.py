import math

import numpy as np


def create_jet_datasets(hh_ds, qcd_ds, args):
    """Create jet-level datasets from event-level HH/QCD datasets."""
    hh_ds = hh_ds.map(
        lambda batch: jets_from_event_batch(batch, is_signal=True, args=args),
        batched=True,
        remove_columns=hh_ds.column_names,
    )

    qcd_ds = qcd_ds.map(
        lambda batch: jets_from_event_batch(batch, is_signal=False, args=args),
        batched=True,
        remove_columns=qcd_ds.column_names,
    )

    return hh_ds, qcd_ds


def delta_r(eta1, phi1, eta2, phi2):
    """Compute angular distance dR between two objects."""
    dphi = abs(phi1 - phi2)
    if dphi > math.pi:
        dphi = 2 * math.pi - dphi
    deta = eta1 - eta2
    return math.sqrt(deta * deta + dphi * dphi)


def select_pf_in_jet(pf_eta, pf_phi, jet_eta, jet_phi, dr_max):
    """Return indices of PF candidates inside a dR cone around the jet."""
    idx = []
    for i in range(len(pf_eta)):
        if delta_r(pf_eta[i], pf_phi[i], jet_eta, jet_phi) < dr_max:
            idx.append(i)
    return idx


def is_hbb_jet(x, jet_idx, dr_max=0.8):
    """Truth-matching logic: checks if a reconstructed jet is Higgs->bb."""
    jet_etas = x["FullReco_GenJetAK8_Eta"]
    jet_phis = x["FullReco_GenJetAK8_Phi"]

    if jet_idx >= len(jet_etas):
        return False

    gen_pid = x["FullReco_GenPart_PID"]
    if not gen_pid:
        return False

    gen_eta = x["FullReco_GenPart_Eta"]
    gen_phi = x["FullReco_GenPart_Phi"]
    d1 = x["FullReco_GenPart_D1"]
    d2 = x["FullReco_GenPart_D2"]

    jet_eta = jet_etas[jet_idx]
    jet_phi = jet_phis[jet_idx]

    n_gen = len(gen_pid)

    for i, pid in enumerate(gen_pid):
        if pid != 25:
            continue

        i1, i2 = d1[i], d2[i]
        if not (0 <= i1 < n_gen and 0 <= i2 < n_gen):
            continue

        if abs(gen_pid[i1]) != 5 or abs(gen_pid[i2]) != 5:
            continue

        dr1 = delta_r(gen_eta[i1], gen_phi[i1], jet_eta, jet_phi)
        dr2 = delta_r(gen_eta[i2], gen_phi[i2], jet_eta, jet_phi)
        if dr1 < dr_max and dr2 < dr_max:
            return True

    return False


def jets_from_event_batch(batch, is_signal, args):
    """Convert one event batch into jet candidates with fixed-size PF inputs."""
    max_pf, l1_features = args
    x_l1s, masks_l1, ys = [], [], []
    jet_pts, jet_masses, jet_etas_out, jet_phis_out, n_pfs_out = [], [], [], [], []

    n_events = len(batch["FullReco_GenJetAK8_Eta"])

    for ievt in range(n_events):
        evt = {k: batch[k][ievt] for k in batch}
        jet_etas = evt["FullReco_GenJetAK8_Eta"]
        jet_phis = evt["FullReco_GenJetAK8_Phi"]

        for j in range(len(jet_etas)):
            jet_eta, jet_phi = float(jet_etas[j]), float(jet_phis[j])
            label = int(is_signal and is_hbb_jet(evt, j))

            idx_l1 = select_pf_in_jet(
                evt["L1T_PFCand_Eta"],
                evt["L1T_PFCand_Phi"],
                jet_eta,
                jet_phi,
                dr_max=0.8,
            )
            if len(idx_l1) == 0:
                continue

            idx_l1 = idx_l1[:max_pf]

            pts = np.array([float(evt["L1T_PFCand_PT"][i]) for i in idx_l1])
            etas = np.array([float(evt["L1T_PFCand_Eta"][i]) for i in idx_l1])
            phis = np.array([float(evt["L1T_PFCand_Phi"][i]) for i in idx_l1])
            ms = np.array([float(evt["L1T_PFCand_Mass"][i]) for i in idx_l1])

            keep = (pts > 15) & (np.abs(etas) < 2.4)
            if not np.any(keep):
                continue

            filtered_indices = [idx_l1[i] for i, keep_i in enumerate(keep) if keep_i]

            pts, etas, phis, ms = (
                pts[keep],
                etas[keep],
                phis[keep],
                ms[keep],
            )
            l1_feats = np.array(
                [[float(evt[k][i]) for k in l1_features] for i in filtered_indices]
            )

            order = np.argsort(-pts)
            pts, etas, phis, ms, l1_feats = (
                pts[order],
                etas[order],
                phis[order],
                ms[order],
                l1_feats[order],
            )

            n = len(l1_feats)
            mask_l1 = np.pad(np.ones(n, dtype=int), (0, max_pf - n), constant_values=0)
            l1_feats = np.pad(l1_feats, ((0, max_pf - n), (0, 0)), mode="constant")

            px = np.sum(pts * np.cos(phis))
            py = np.sum(pts * np.sin(phis))
            pz = np.sum(pts * np.sinh(etas))
            e = np.sum(np.sqrt((pts * np.cosh(etas)) ** 2 + ms**2))

            jet_pt = np.sqrt(px**2 + py**2)
            jet_mass = np.sqrt(max(e**2 - (px**2 + py**2 + pz**2), 0.0))

            if jet_pt >= 150:
                x_l1s.append(l1_feats)
                masks_l1.append(mask_l1)
                ys.append([1.0, 0.0] if label else [0.0, 1.0])
                jet_pts.append(float(jet_pt))
                jet_masses.append(float(jet_mass))
                jet_etas_out.append(jet_eta)
                jet_phis_out.append(jet_phi)
                n_pfs_out.append(int(len(idx_l1)))

    return {
        "x": x_l1s,
        "mask": masks_l1,
        "label": ys,
        "jet_pt": jet_pts,
        "jet_mass": jet_masses,
        "jet_eta": jet_etas_out,
        "jet_phi": jet_phis_out,
        "n_pf": n_pfs_out,
    }
