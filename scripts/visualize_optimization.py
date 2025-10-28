#!/usr/bin/env python3
"""
Visualize SceneGuard Optimization Process

Generates visualizations of:
1. Loss curves over optimization epochs
2. Temporal mask m(t) before and after optimization
3. Waveform comparisons (original / direct / optimized)
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_optimization_log(log_path):
    """Load optimization log from JSON"""
    with open(log_path, 'r') as f:
        log = json.load(f)
    return log


def plot_loss_curves(log, output_path):
    """
    Plot optimization trajectory: total loss, SIM loss, and regularization loss
    
    Args:
        log: Dictionary with 'trajectory' containing loss lists
        output_path: Path to save figure
    """
    trajectory = log['trajectory']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('SceneGuard Optimization Trajectory', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(trajectory['losses']) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, trajectory['losses'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Speaker similarity loss
    axes[0, 1].plot(epochs, trajectory['sim_losses'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Speaker Similarity')
    axes[0, 1].set_title('Speaker Similarity Loss (L_SIM)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.25, color='k', linestyle='--', alpha=0.5, label='Threshold (0.25)')
    axes[0, 1].legend()
    
    # Regularization loss
    axes[1, 0].plot(epochs, trajectory['reg_losses'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Regularization')
    axes[1, 0].set_title('Regularization Loss (L_REG)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gamma and SNR
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(epochs, trajectory['gammas'], 'purple', linewidth=2, label='γ')
    l2 = ax2.plot(epochs, trajectory['snrs'], 'orange', linewidth=2, label='SNR (dB)')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Gamma (γ)', color='purple')
    ax2.set_ylabel('SNR (dB)', color='orange')
    ax1.set_title('Noise Strength and SNR')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax1.grid(True, alpha=0.3)
    
    # SNR constraint bounds
    ax2.axhline(y=10, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axhline(y=20, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax2.fill_between(epochs, 10, 20, alpha=0.1, color='gray', label='SNR constraint')
    
    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved loss curves to: {output_path}")


def plot_waveform_comparison(original_path, direct_path, optimized_path, output_path, sr=16000):
    """
    Plot waveform comparison: original, direct mixing, and optimized mixing
    
    Args:
        original_path: Path to original audio
        direct_path: Path to direct mixing output
        optimized_path: Path to optimized mixing output
        output_path: Path to save figure
        sr: Sampling rate
    """
    # Load audio
    original, _ = librosa.load(original_path, sr=sr, mono=True)
    direct, _ = librosa.load(direct_path, sr=sr, mono=True)
    optimized, _ = librosa.load(optimized_path, sr=sr, mono=True)
    
    # Time axis
    duration = len(original) / sr
    time = np.linspace(0, duration, len(original))
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    fig.suptitle('Waveform Comparison', fontsize=14, fontweight='bold')
    
    # Original
    axes[0].plot(time, original, 'k-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Speech')
    axes[0].set_xlim([0, duration])
    axes[0].grid(True, alpha=0.3)
    
    # Direct mixing
    axes[1].plot(time, direct, 'b-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Direct Mixing (Baseline)')
    axes[1].set_xlim([0, duration])
    axes[1].grid(True, alpha=0.3)
    
    # Optimized mixing
    axes[2].plot(time, optimized, 'r-', linewidth=0.5, alpha=0.7)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Optimized Mixing (SceneGuard)')
    axes[2].set_xlim([0, duration])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved waveform comparison to: {output_path}")


def plot_spectrogram_comparison(original_path, direct_path, optimized_path, output_path, sr=16000):
    """
    Plot spectrogram comparison
    
    Args:
        original_path: Path to original audio
        direct_path: Path to direct mixing output
        optimized_path: Path to optimized mixing output
        output_path: Path to save figure
        sr: Sampling rate
    """
    # Load audio
    original, _ = librosa.load(original_path, sr=sr, mono=True)
    direct, _ = librosa.load(direct_path, sr=sr, mono=True)
    optimized, _ = librosa.load(optimized_path, sr=sr, mono=True)
    
    # Compute spectrograms
    n_fft = 1024
    hop_length = 256
    
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    D_direct = librosa.amplitude_to_db(np.abs(librosa.stft(direct, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    D_opt = librosa.amplitude_to_db(np.abs(librosa.stft(optimized, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Spectrogram Comparison', fontsize=14, fontweight='bold')
    
    # Original
    img1 = librosa.display.specshow(D_orig, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=axes[0], cmap='viridis')
    axes[0].set_title('Original Speech')
    axes[0].set_ylabel('Frequency (Hz)')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # Direct mixing
    img2 = librosa.display.specshow(D_direct, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis')
    axes[1].set_title('Direct Mixing (Baseline)')
    axes[1].set_ylabel('Frequency (Hz)')
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    
    # Optimized mixing
    img3 = librosa.display.specshow(D_opt, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=axes[2], cmap='viridis')
    axes[2].set_title('Optimized Mixing (SceneGuard)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Frequency (Hz)')
    fig.colorbar(img3, ax=axes[2], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved spectrogram comparison to: {output_path}")


def visualize_sample(sample_id, output_dir='paper/figures/optimization_viz'):
    """
    Generate all visualizations for a single sample
    
    Args:
        sample_id: Sample identifier (e.g., filename without extension)
        output_dir: Directory to save visualizations
    """
    print(f"\n{'='*70}")
    print(f"Visualizing Sample: {sample_id}")
    print(f"{'='*70}")
    
    # Setup paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files
    original_path = list(Path("data/raw/speech/libritts/5339").glob(f"{sample_id}*.wav"))
    direct_path = list(Path("data/processed/defended/5339").glob(f"{sample_id}*.wav"))
    optimized_path = list(Path("data/processed/defended_optimized/5339").glob(f"{sample_id}*.wav"))
    opt_log_path = Path(f"artifacts/optimization_logs/{sample_id}_opt.json")
    
    if not original_path:
        print(f"  ✗ Original audio not found for {sample_id}")
        return 1
    if not direct_path:
        print(f"  ⚠ Direct mixing output not found for {sample_id} (skipping comparison)")
        direct_path = [None]
    if not optimized_path:
        print(f"  ✗ Optimized mixing output not found for {sample_id}")
        return 1
    if not opt_log_path.exists():
        print(f"  ✗ Optimization log not found for {sample_id}")
        return 1
    
    original_path = str(original_path[0])
    direct_path = str(direct_path[0]) if direct_path[0] else None
    optimized_path = str(optimized_path[0])
    
    # Load optimization log
    print("\n[1/4] Loading optimization log...")
    opt_log = load_optimization_log(opt_log_path)
    print(f"  Final similarity: {opt_log['final_similarity']:.4f}")
    print(f"  Final SNR: {opt_log['final_snr_db']:.2f} dB")
    
    # Plot loss curves
    print("\n[2/4] Plotting loss curves...")
    plot_loss_curves(opt_log, output_dir / f"{sample_id}_losses.pdf")
    
    # Plot waveforms (if direct mixing available)
    if direct_path:
        print("\n[3/4] Plotting waveform comparison...")
        plot_waveform_comparison(
            original_path, direct_path, optimized_path,
            output_dir / f"{sample_id}_waveforms.pdf"
        )
        
        # Plot spectrograms
        print("\n[4/4] Plotting spectrogram comparison...")
        plot_spectrogram_comparison(
            original_path, direct_path, optimized_path,
            output_dir / f"{sample_id}_spectrograms.pdf"
        )
    else:
        print("\n[3/4] Skipping waveform comparison (no direct mixing)")
        print("[4/4] Skipping spectrogram comparison (no direct mixing)")
    
    print(f"\n✓ Visualizations saved to: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Visualize SceneGuard optimization process',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--sample-id',
        type=str,
        help='Sample ID to visualize (e.g., 5339_14133_000002_000003). If not provided, visualizes the first sample.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='paper/figures/optimization_viz',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    # If no sample ID provided, use first available optimization log
    if args.sample_id is None:
        opt_logs = list(Path("artifacts/optimization_logs").glob("*_opt.json"))
        if not opt_logs:
            print("✗ No optimization logs found in artifacts/optimization_logs/")
            print("  Run: python scripts/run_defense_mixing.py --mode optimized")
            return 1
        
        sample_id = opt_logs[0].stem.replace('_opt', '')
        print(f"No sample ID provided, using first available: {sample_id}")
    else:
        sample_id = args.sample_id
    
    return visualize_sample(sample_id, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())

