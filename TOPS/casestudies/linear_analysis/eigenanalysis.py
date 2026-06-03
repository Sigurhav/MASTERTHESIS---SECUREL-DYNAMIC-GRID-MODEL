import src.dynamic as dps
import src.linear_ps as dps_mdl
import plotting as sim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def analyze_eigenvalues(ps_lin, ps):
    """
    Simplified eigenvalue analysis with clear output
    """

    # Get eigenvalues and participation factors
    eigs = ps_lin.eigs
    pfs = ps_lin.lev.T * ps_lin.rev  # Participation factors
    pfs_abs = np.abs(pfs) / np.max(np.abs(pfs), axis=0)  # Normalized

    # Create results dictionary
    results = {
        'eigenvalues': [],
        'participation': []
    }

    print("\n" + "=" * 80)
    print("EIGENVALUE ANALYSIS - SIMPLIFIED REPORT")
    print("=" * 80)

    # 1. EIGENVALUE TABLE
    print("\n### ALLE EGENVERDIER (MODES) ###\n")
    print(f"{'Mode':<6} {'Eigenverdi':<25} {'Frekvens':<12} {'Demping (ζ)':<12} {'Type':<15}")
    print("-" * 80)

    for i, eig in enumerate(eigs):
        if abs(eig.imag) > 0.01:  # Complex eigenvalue
            freq_hz = abs(eig.imag) / (2 * np.pi)
            damping_ratio = -eig.real / abs(eig)
            mode_type = "Oscillerende"
            eig_str = f"{eig.real:7.3f} ± {abs(eig.imag):6.3f}j"
            results['eigenvalues'].append({
                'Mode': i,
                'Eigenvalue': eig,
                'Real': eig.real,
                'Imag': eig.imag,
                'Frequency_Hz': freq_hz,
                'Damping_ratio': damping_ratio,
                'Type': mode_type
            })
            print(f"{i:<6} {eig_str:<25} {freq_hz:6.3f} Hz   {damping_ratio:6.4f}      {mode_type:<15}")
        else:  # Real eigenvalue
            mode_type = "Reell (ikke-osc.)"
            eig_str = f"{eig.real:7.3f}"
            results['eigenvalues'].append({
                'Mode': i,
                'Eigenvalue': eig,
                'Real': eig.real,
                'Imag': 0,
                'Frequency_Hz': 0,
                'Damping_ratio': np.inf,
                'Type': mode_type
            })
            print(f"{i:<6} {eig_str:<25} {'N/A':<12} {'N/A':<12} {mode_type:<15}")

    # 2. IDENTIFY CRITICAL MODES
    print("\n### KRITISKE MODER (dårlig demping) ###\n")
    critical_modes = []
    for res in results['eigenvalues']:
        if res['Type'] == "Oscillerende" and res['Damping_ratio'] < 0.05:
            critical_modes.append(res)
            print(f"⚠️  Mode {res['Mode']}: ζ = {res['Damping_ratio']:.4f} (< 0.05) - DÅRLIG DEMPING!")
            print(f"    Frekvens: {res['Frequency_Hz']:.3f} Hz")
            print(f"    Eigenverdi: {res['Real']:.3f} ± {abs(res['Imag']):.3f}j\n")

    if not critical_modes:
        print("✓ Ingen kritiske moder funnet (alle har ζ > 0.05)")

    # 3. PARTICIPATION FACTOR ANALYSIS
    print("\n### PARTICIPATION FACTORS - HVEM DELTAR I HVER MODE? ###\n")

    # For each mode, show top participating states
    for i, eig in enumerate(eigs):
        mode_pfs = pfs_abs[:, i]
        top_indices = np.argsort(mode_pfs)[::-1][:5]  # Top 5 states

        print(f"\nMode {i}: {results['eigenvalues'][i]['Type']}")
        if results['eigenvalues'][i]['Type'] == "Oscillerende":
            print(
                f"  Frekvens: {results['eigenvalues'][i]['Frequency_Hz']:.3f} Hz, ζ = {results['eigenvalues'][i]['Damping_ratio']:.4f}")
        else:
            print(f"  Eigenverdi: {eig.real:.3f}")

        print("  Top deltakere:")
        for idx in top_indices:
            if mode_pfs[idx] > 0.01:  # Only show if participation > 1%
                state_desc = ps.state_desc[idx]
                print(f"    - {state_desc}: {mode_pfs[idx]:.3f} (deltakelsefaktor)")

    # 4. STATE-BY-STATE ANALYSIS
    print("\n\n### HVILKE MODER PÅVIRKES AV HVER STATE? ###\n")

    state_categories = {
        'Governor (GOV1)': [i for i, desc in enumerate(ps.state_desc) if 'GOV1' in str(desc)],
        'AVR': [i for i, desc in enumerate(ps.state_desc) if 'AVR' in str(desc)],
        'Generator (G1)': [i for i, desc in enumerate(ps.state_desc) if 'G1' in str(desc)],
        'Infinite Bus (IB)': [i for i, desc in enumerate(ps.state_desc) if 'IB' in str(desc)]
    }

    for category, state_indices in state_categories.items():
        print(f"\n{category}:")
        for state_idx in state_indices:
            state_desc = ps.state_desc[state_idx]
            print(f"  {state_desc}:")

            # Find modes where this state participates significantly
            participating_modes = []
            for mode_idx in range(len(eigs)):
                if pfs_abs[state_idx, mode_idx] > 0.3:  # Significant participation
                    participating_modes.append({
                        'mode': mode_idx,
                        'pf': pfs_abs[state_idx, mode_idx],
                        'type': results['eigenvalues'][mode_idx]['Type']
                    })

            if participating_modes:
                for pm in participating_modes:
                    print(f"    → Mode {pm['mode']}: {pm['type']} (deltakelse: {pm['pf']:.3f})")
            else:
                print(f"    → Lav deltakelse i alle moder")

    # 5. CREATE VISUALIZATION
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All eigenvalues
    ax = axes[0, 0]
    ax.scatter(np.array([e['Real'] for e in results['eigenvalues']]),
               np.array([e['Imag'] for e in results['eigenvalues']]),
               c='blue', s=100, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Real del (σ)', fontsize=12)
    ax.set_ylabel('Imaginær del (ω)', fontsize=12)
    ax.set_title('Alle egenverdier', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate critical modes
    for cm in critical_modes:
        ax.scatter(cm['Real'], cm['Imag'], c='red', s=200, marker='*',
                   edgecolors='black', linewidths=2, zorder=5)
        ax.annotate(f"Mode {cm['Mode']}\nζ={cm['Damping_ratio']:.3f}",
                    (cm['Real'], cm['Imag']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    fontsize=9)

    # Plot 2: Participation factors heatmap for critical modes
    ax = axes[0, 1]
    if critical_modes:
        critical_mode_indices = [cm['Mode'] for cm in critical_modes]
        pf_data = pfs_abs[:, critical_mode_indices]

        im = ax.imshow(pf_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_yticks(range(len(ps.state_desc)))
        ax.set_yticklabels([str(desc) for desc in ps.state_desc], fontsize=8)
        ax.set_xticks(range(len(critical_mode_indices)))
        ax.set_xticklabels([f"Mode {i}" for i in critical_mode_indices], fontsize=9)
        ax.set_title('Deltakelsefaktorer for kritiske moder', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Deltakelse')
    else:
        ax.text(0.5, 0.5, 'Ingen kritiske moder', ha='center', va='center', fontsize=14)
        ax.set_title('Deltakelsefaktorer for kritiske moder', fontsize=12, fontweight='bold')

    # Plot 3: Speed participation
    ax = axes[1, 0]
    speed_indices = [i for i, desc in enumerate(ps.state_desc) if 'speed' in str(desc).lower()]
    if speed_indices:
        max_speed_pf = np.max(pfs_abs[speed_indices, :], axis=0)
        colors = ['red' if pf > 0.5 else 'blue' for pf in max_speed_pf]
        ax.scatter(np.array([e['Real'] for e in results['eigenvalues']]),
                   np.array([e['Imag'] for e in results['eigenvalues']]),
                   c=colors, s=100, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Real del (σ)', fontsize=12)
        ax.set_ylabel('Imaginær del (ω)', fontsize=12)
        ax.set_title('Moder med SPEED deltakelse (rød = høy)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Plot 4: Governor participation
    ax = axes[1, 1]
    gov_indices = [i for i, desc in enumerate(ps.state_desc) if 'GOV' in str(desc)]
    if gov_indices:
        max_gov_pf = np.max(pfs_abs[gov_indices, :], axis=0)
        colors = ['green' if pf > 0.5 else 'blue' for pf in max_gov_pf]
        ax.scatter(np.array([e['Real'] for e in results['eigenvalues']]),
                   np.array([e['Imag'] for e in results['eigenvalues']]),
                   c=colors, s=100, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Real del (σ)', fontsize=12)
        ax.set_ylabel('Imaginær del (ω)', fontsize=12)
        ax.set_title('Moder med GOVERNOR deltakelse (grønn = høy)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return results, fig


def main():
    import casestudies.ps_data.sixtyMWimport as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    results, fig = analyze_eigenvalues(ps_lin, ps)

    # Save figure
    #fig.savefig('/mnt/user-data/outputs/eigenvalue_analysis.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 80)
    print("Analyse fullført! Plot lagret som 'eigenvalue_analysis.png'")
    print("=" * 80)

    plt.show()


if __name__ == '__main__':
    main()