import numpy as np
import cv2
import os
import argparse

def _safe_alpha(index, total):
    if total <= 1:
        return 0.0
    return index / float(total - 1)


def _ensure_odd_kernel(k_size):
    k_size = max(1, int(k_size))
    if k_size % 2 == 0:
        k_size += 1
    return k_size


def _prepare_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)

    # Common exported shape from torch is (1, H, W); squeeze singleton dims only.
    if depth.ndim == 3:
        depth = np.squeeze(depth)

    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D after squeeze, got shape {depth.shape}")

    finite_mask = np.isfinite(depth)
    if not np.any(finite_mask):
        raise ValueError("Depth map has no finite values")

    finite_vals = depth[finite_mask]
    fill_value = float(np.median(finite_vals))
    depth = np.where(finite_mask, depth, fill_value)

    # Clip extreme outliers so CoC does not explode on sparse artifacts.
    p01 = float(np.percentile(finite_vals, 1.0))
    p99 = float(np.percentile(finite_vals, 99.0))
    if p99 > p01:
        depth = np.clip(depth, p01, p99)

    return depth, finite_vals


def _build_blur_layers(rgb, num_layers=5, max_kernel=31):
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    rgb_f32 = rgb.astype(np.float32)
    max_kernel = _ensure_odd_kernel(max_kernel)

    if num_layers == 1:
        return [rgb_f32]

    blurred_layers = []
    for i in range(num_layers):
        if i == 0:
            blurred_layers.append(rgb_f32)
            continue

        k_size = int(round((i / float(num_layers - 1)) * max_kernel))
        k_size = _ensure_odd_kernel(k_size)
        blurred = cv2.GaussianBlur(rgb_f32, (k_size, k_size), 0)
        blurred_layers.append(blurred)

    return blurred_layers


def apply_dof(
    rgb,
    depth,
    focus_depth,
    aperture=5.0,
    num_layers=5,
    max_kernel=31,
    blurred_layers=None,
    depth_span=None,
):
    """
    Applies depth of field to an image using pre-computed blur layers.
    
    Args:
        rgb: Input RGB image (H, W, 3).
        depth: Input Depth map (H, W).
        focus_depth: The Z-value to keep in focus.
        aperture: Multiplier for the blur intensity.
        num_layers: Number of discrete blur levels to compute.
        max_kernel: Maximum size of the Gaussian blur kernel.
    """
    if blurred_layers is None:
        blurred_layers = _build_blur_layers(rgb, num_layers=num_layers, max_kernel=max_kernel)
    else:
        num_layers = len(blurred_layers)

    # Calculate the Circle of Confusion (CoC) scaled by depth span.
    if depth_span is None:
        finite_mask = np.isfinite(depth)
        finite_vals = depth[finite_mask]
        p02 = float(np.percentile(finite_vals, 2.0))
        p98 = float(np.percentile(finite_vals, 98.0))
        depth_span = max(p98 - p02, 1e-6)

    coc = (np.abs(depth - focus_depth) / float(depth_span)) * float(aperture) * (num_layers - 1)

    # Normalize CoC to range [0, num_layers - 1]
    coc_normalized = np.clip(coc, 0, num_layers - 1)
    
    # Interpolate between layers based on CoC
    output = np.zeros_like(blurred_layers[0], dtype=np.float32)
    
    # Lower integer bound of the layer
    layer_idx_lower = np.floor(coc_normalized).astype(np.int32)
    # Upper integer bound of the layer
    layer_idx_upper = np.clip(layer_idx_lower + 1, 0, num_layers - 1)
    
    # Interpolation weights
    weight_upper = coc_normalized - layer_idx_lower
    weight_lower = 1.0 - weight_upper
    
    # Construct final image
    # Note: Vectorized approach avoiding explicit python loops over pixels
    for i in range(num_layers):
        mask_lower = (layer_idx_lower == i)[..., np.newaxis]
        mask_upper = (layer_idx_upper == i)[..., np.newaxis]
        
        output += blurred_layers[i] * weight_lower[..., np.newaxis] * mask_lower
        output += blurred_layers[i] * weight_upper[..., np.newaxis] * mask_upper
        
    return np.clip(output, 0, 255).astype(np.uint8)

def generate_dof_animation(
    rgb_path,
    depth_path,
    out_dir,
    fg_depth=None,
    bg_depth=None,
    frames=90,
    max_aperture=5.0,
    num_layers=5,
    max_kernel=31,
    fg_percentile=10.0,
    bg_percentile=90.0,
):
    """
    Generates the animation frames transitioning the focus.
    """
    if frames < 1:
        raise ValueError("frames must be >= 1")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    rgb_base = cv2.imread(rgb_path)
    if rgb_base is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")

    depth_base = np.load(depth_path)
    depth_base, finite_depth_vals = _prepare_depth(depth_base)

    # Reuse the same blur pyramid for all frames (much faster).
    blur_layers = _build_blur_layers(rgb_base, num_layers=num_layers, max_kernel=max_kernel)

    depth_near = float(np.percentile(finite_depth_vals, 2.0))
    depth_far = float(np.percentile(finite_depth_vals, 98.0))
    depth_span = max(depth_far - depth_near, 1e-6)

    if fg_depth is None:
        fg_depth = float(np.percentile(finite_depth_vals, fg_percentile))
    if bg_depth is None:
        bg_depth = float(np.percentile(finite_depth_vals, bg_percentile))

    print(
        f"Using focus depths: fg={fg_depth:.6f}, bg={bg_depth:.6f} "
        f"(p{fg_percentile:.1f}/p{bg_percentile:.1f})"
    )

    # Phase 1: Normal Gaussian Splatting (No DoF)
    # For the first few frames, we save the original image
    phase1_frames = int(round(frames * 0.2))
    phase1_frames = max(0, min(phase1_frames, frames))
    for i in range(phase1_frames):
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), rgb_base)
        print(f"Generated frame {i} (Phase 1: No DoF)")

    # Phase 2: Transition from infinity (or no blur) to foreground focus
    phase2_frames = int(round(frames * 0.4))
    phase2_frames = max(0, min(phase2_frames, frames - phase1_frames))
    # Start focus past the far percentile to avoid scale-dependent hardcoded offsets.
    start_focus = max(float(fg_depth), depth_far + 0.25 * depth_span)
    
    for i in range(phase2_frames):
        alpha = _safe_alpha(i, phase2_frames)
        # Interpolate focus depth
        current_focus = start_focus * (1 - alpha) + fg_depth * alpha
        
        # Increase aperture gradually to smoothly introduce the blur
        current_aperture = float(max_aperture) * alpha
        
        out_img = apply_dof(
            rgb_base,
            depth_base,
            current_focus,
            aperture=current_aperture,
            num_layers=num_layers,
            max_kernel=max_kernel,
            blurred_layers=blur_layers,
            depth_span=depth_span,
        )
        frame_idx = phase1_frames + i
        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"), out_img)
        print(f"Generated frame {frame_idx} (Phase 2: Focusing on FG)")

    # Phase 3: Transition from foreground to background focus
    phase3_frames = frames - phase1_frames - phase2_frames
    for i in range(phase3_frames):
        alpha = _safe_alpha(i, phase3_frames)
        # Smooth interpolation (ease-in-out could be used here instead of linear)
        current_focus = fg_depth * (1 - alpha) + bg_depth * alpha

        out_img = apply_dof(
            rgb_base,
            depth_base,
            current_focus,
            aperture=float(max_aperture),
            num_layers=num_layers,
            max_kernel=max_kernel,
            blurred_layers=blur_layers,
            depth_span=depth_span,
        )
        frame_idx = phase1_frames + phase2_frames + i
        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"), out_img)
        print(f"Generated frame {frame_idx} (Phase 3: Focusing on BG)")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a depth-of-field rack-focus animation (foreground to background)."
    )
    parser.add_argument("--rgb", required=True, help="Path to the RGB input image (e.g., 00000.png)")
    parser.add_argument("--depth", required=True, help="Path to the depth .npy file (e.g., 00000.npy)")
    parser.add_argument("--out", required=True, help="Output directory for generated frames")
    parser.add_argument("--fg-depth", type=float, default=None, help="Depth value of foreground focus")
    parser.add_argument("--bg-depth", type=float, default=None, help="Depth value of background focus")
    parser.add_argument(
        "--fg-percentile",
        type=float,
        default=10.0,
        help="Percentile used for auto foreground depth (default: 10)",
    )
    parser.add_argument(
        "--bg-percentile",
        type=float,
        default=90.0,
        help="Percentile used for auto background depth (default: 90)",
    )
    parser.add_argument("--frames", type=int, default=90, help="Total number of frames (default: 90)")
    parser.add_argument(
        "--max-aperture",
        type=float,
        default=5.0,
        help="Maximum blur strength used during rack focus (default: 5.0)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=5,
        help="Number of discrete blur layers (default: 5)",
    )
    parser.add_argument(
        "--max-kernel",
        type=int,
        default=31,
        help="Maximum Gaussian kernel size (odd preferred, default: 31)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    generate_dof_animation(
        rgb_path=cli_args.rgb,
        depth_path=cli_args.depth,
        out_dir=cli_args.out,
        fg_depth=cli_args.fg_depth,
        bg_depth=cli_args.bg_depth,
        frames=cli_args.frames,
        max_aperture=cli_args.max_aperture,
        num_layers=cli_args.num_layers,
        max_kernel=cli_args.max_kernel,
        fg_percentile=cli_args.fg_percentile,
        bg_percentile=cli_args.bg_percentile,
    )

# Example execution parameters:
# fg_depth = 2.5 # Check your depth.npy values to find the exact distance of the foreground object
# bg_depth = 8.0 # Check your depth.npy values to find the background distance
# generate_dof_animation("frames/rgb_0000.png", "frames/depth_0000.npy", "output_anim", fg_depth, bg_depth)