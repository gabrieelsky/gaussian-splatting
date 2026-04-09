import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from utils.sh_utils import eval_sh


def evaluate_sh(f_dc, f_rest, points, c2w):
    num_points = points.shape[0]
    sh_rest = f_rest.reshape(num_points, 3, -1)
    sh_coeffs = torch.cat([f_dc.unsqueeze(-1), sh_rest], dim=-1)

    camera_center = c2w[:3, 3].unsqueeze(0)
    view_dir = points - camera_center
    view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)

    sh_degree = int(np.sqrt(sh_coeffs.shape[-1])) - 1
    rgb = eval_sh(sh_degree, sh_coeffs, view_dir)
    return torch.clamp_min(rgb + 0.5, 0.0)


def project_points(pc, c2w, fx, fy, cx, cy):
    w2c = torch.eye(4, device=pc.device)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c[:3, :3] = R.t()
    w2c[:3, 3] = -R.t() @ t

    pc_h = torch.cat([pc, torch.ones_like(pc[:, :1])], dim=1)
    PC = ((w2c @ pc_h.t()).t())[:, :3]
    x, y, z = PC[:, 0], PC[:, 1], PC[:, 2]

    uv = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)
    return uv, x, y, z


def inv2x2(M, eps=1e-12):
    a = M[:, 0, 0]
    b = M[:, 0, 1]
    c = M[:, 1, 0]
    d = M[:, 1, 1]
    det = a * d - b * c
    safe_det = torch.clamp(det, min=eps)
    inv = torch.empty_like(M)
    inv[:, 0, 0] = d / safe_det
    inv[:, 0, 1] = -b / safe_det
    inv[:, 1, 0] = -c / safe_det
    inv[:, 1, 1] = a / safe_det
    return inv

def quat_to_rotmat(quat):
    r, x, y, z = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xr, yr, zr = x * r, y * r, z * r

    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - zr), 2 * (xz + yr),
        2 * (xy + zr), 1 - 2 * (xx + zz), 2 * (yz - xr),
        2 * (xz - yr), 2 * (yz + xr), 1 - 2 * (xx + yy)
    ], dim=-1).reshape(quat.shape[:-1] + (3, 3))
    return R

def build_sigma_from_params(scale_raw, q_raw):
    scale = torch.exp(scale_raw).clamp_min(1e-6)
    q = q_raw / (q_raw.norm(dim=-1, keepdim=True) + 1e-9)
    R = quat_to_rotmat(q)
    S = torch.diag_embed(scale)
    return R @ S @ S @ R.transpose(1, 2)


def scale_intrinsics(H, W, H_src, W_src, fx, fy, cx, cy):
    scale_x = W / W_src
    scale_y = H / H_src
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y
    return fx_scaled, fy_scaled, cx_scaled, cy_scaled


def load_gaussians_from_ply(ply_path, device):
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    v = ply["vertex"]

    pos = torch.tensor(np.stack([v["x"], v["y"], v["z"]], axis=1), dtype=torch.float32, device=device)
    opacity_raw = torch.tensor(np.asarray(v["opacity"]).reshape(-1, 1), dtype=torch.float32, device=device)
    scale_raw = torch.tensor(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1), dtype=torch.float32, device=device)
    q_raw = torch.tensor(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1), dtype=torch.float32, device=device)

    f_dc_keys = sorted([k for k in v.data.dtype.names if k.startswith("f_dc_")], key=lambda s: int(s.split("_")[-1]))
    f_rest_keys = sorted([k for k in v.data.dtype.names if k.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    f_dc = torch.tensor(np.stack([v[k] for k in f_dc_keys], axis=1), dtype=torch.float32, device=device)
    f_rest = torch.tensor(np.stack([v[k] for k in f_rest_keys], axis=1), dtype=torch.float32, device=device)

    return pos, opacity_raw, f_dc, f_rest, scale_raw, q_raw


def camera_info_to_c2w(cam_info, device):
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = cam_info.R.transpose()
    Rt[:3, 3] = cam_info.T
    c2w = np.linalg.inv(Rt)
    return torch.tensor(c2w, dtype=torch.float32, device=device)


def resolve_output_dir(args):
    base_output_dir = Path(args.output_dir)

    # Keep outputs separated by scene/model to avoid overwriting existing renders.
    scene_name = None
    if args.model_path:
        scene_name = Path(args.model_path).name
    elif args.trajectory_file:
        scene_name = Path(args.trajectory_file).stem

    return base_output_dir / scene_name if scene_name else base_output_dir


@torch.no_grad()
def render(
    pos,
    color,
    opacity_raw,
    sigma,
    c2w,
    H,
    W,
    fx,
    fy,
    cx,
    cy,
    near=2e-3,
    far=100,
    pix_guard=64,
    T=16,
    min_conis=1e-6,
    chi_square_clip=9.21, 
    alpha_max=0.99, 
    alpha_cutoff=1/255.,
    antialiasing=True, h_var=0.3, bg_color=None):

    if bg_color is None:
        bg = torch.zeros((3,), device=pos.device, dtype=pos.dtype)
    else:
        bg = torch.as_tensor(bg_color, device=pos.device, dtype=pos.dtype).reshape(3)

    uv, x, y, z = project_points(pos, c2w, fx, fy, cx, cy)
    in_guard = (uv[:, 0] > -pix_guard) & (uv[:, 0] < W + pix_guard) & (uv[:, 1] > -pix_guard) & (uv[:, 1] < H + pix_guard) & (z > near) & (z < far)

    uv = uv[in_guard]
    color = color[in_guard]
    opacity = torch.sigmoid(opacity_raw[in_guard]).clamp(0, 0.999)
    if opacity.ndim == 2 and opacity.shape[-1] == 1:
        opacity = opacity.squeeze(-1)
    z = z[in_guard]
    x = x[in_guard]
    y = y[in_guard]
    sigma = sigma[in_guard]

    Rcw = c2w[:3, :3]
    Rwc = Rcw.t()
    invz = 1 / z.clamp_min(1e-6)
    invz2 = invz * invz
    tanfovx = W / (2.0 * fx)
    tanfovy = H / (2.0 * fy)
    limx = 1.3 * tanfovx
    limy = 1.3 * tanfovy
    txtz = x * invz
    tytz = y * invz
    x = torch.clamp(txtz, min=-limx, max=limx) * z
    y = torch.clamp(tytz, min=-limy, max=limy) * z
    J = torch.zeros((sigma.shape[0], 2, 3), device=sigma.device, dtype=sigma.dtype)
    J[:, 0, 0] = fx * invz
    J[:, 1, 1] = fy * invz
    J[:, 0, 2] = -fx * x * invz2
    J[:, 1, 2] = -fy * y * invz2

    tmp = Rwc.unsqueeze(0) @ sigma @ Rwc.t().unsqueeze(0)
    sigma_camera = J @ tmp @ J.transpose(1, 2)
    sigma_camera = 0.5 * (sigma_camera + sigma_camera.transpose(1, 2))

    evals, evecs = torch.linalg.eigh(sigma_camera)
    evals = torch.clamp(evals, min=1e-6, max=1e4)
    sigma_camera = evecs @ torch.diag_embed(evals) @ evecs.transpose(1, 2)

    det_cov = sigma_camera[:, 0, 0] * sigma_camera[:, 1, 1] - sigma_camera[:, 0, 1] * sigma_camera[:, 1, 0]
    sigma_camera[:, 0, 0] += h_var
    sigma_camera[:, 1, 1] += h_var
    det_cov_plus_h = sigma_camera[:, 0, 0] * sigma_camera[:, 1, 1] - sigma_camera[:, 0, 1] * sigma_camera[:, 1, 0]
    h_convolution_scaling = torch.ones_like(opacity)
    if antialiasing:
        ratio = det_cov / det_cov_plus_h.clamp_min(1e-12)
        h_convolution_scaling = torch.sqrt(torch.clamp(ratio, min=2.5e-5))
    opacity = (opacity * h_convolution_scaling).clamp(0.0, 0.999)

    keep = torch.isfinite(sigma_camera.reshape(sigma.shape[0], -1)).all(dim=-1)
    uv = uv[keep]
    color = color[keep]
    opacity = opacity[keep]
    z = z[keep]
    sigma_camera = sigma_camera[keep]
    evals = evals[keep]

    order = torch.argsort(z, descending=False)
    uv = uv[order]
    u = uv[:, 0]
    v = uv[:, 1]
    color = color[order]
    opacity = opacity[order]
    sigma_camera = sigma_camera[order]
    evals = evals[order]

    major_variance = evals[:, 1].clamp_min(1e-12).clamp_max(1e4)
    radius = torch.ceil(3.0 * torch.sqrt(major_variance)).to(torch.int64)
    umin = torch.floor(u - radius).to(torch.int64)
    umax = torch.floor(u + radius).to(torch.int64)
    vmin = torch.floor(v - radius).to(torch.int64)
    vmax = torch.floor(v + radius).to(torch.int64)

    on_screen = (umax >= 0) & (umin < W) & (vmax >= 0) & (vmin < H)
    if not on_screen.any():
        raise RuntimeError("All projected points are off-screen")

    u, v = u[on_screen], v[on_screen]
    color = color[on_screen]
    opacity = opacity[on_screen]
    sigma_camera = sigma_camera[on_screen]
    umin, umax = umin[on_screen], umax[on_screen]
    vmin, vmax = vmin[on_screen], vmax[on_screen]

    umin = umin.clamp(0, W - 1)
    umax = umax.clamp(0, W - 1)
    vmin = vmin.clamp(0, H - 1)
    vmax = vmax.clamp(0, H - 1)

    umin_tile = (umin // T).to(torch.int64)
    umax_tile = (umax // T).to(torch.int64)
    vmin_tile = (vmin // T).to(torch.int64)
    vmax_tile = (vmax // T).to(torch.int64)

    n_u = umax_tile - umin_tile + 1
    n_v = vmax_tile - vmin_tile + 1
    max_u = int(n_u.max().item())
    max_v = int(n_v.max().item())

    nb_gaussians = umin_tile.shape[0]
    span_indices_u = torch.arange(max_u, device=pos.device, dtype=torch.int64)
    span_indices_v = torch.arange(max_v, device=pos.device, dtype=torch.int64)

    tile_u = (umin_tile[:, None, None] + span_indices_u[None, :, None]).expand(nb_gaussians, max_u, max_v)
    tile_v = (vmin_tile[:, None, None] + span_indices_v[None, None, :]).expand(nb_gaussians, max_u, max_v)

    mask = (span_indices_u[None, :, None] < n_u[:, None, None]) & (span_indices_v[None, None, :] < n_v[:, None, None])
    flat_tile_u = tile_u[mask]
    flat_tile_v = tile_v[mask]

    nb_tiles_per_gaussian = n_u * n_v
    gaussian_ids = torch.repeat_interleave(torch.arange(nb_gaussians, device=pos.device, dtype=torch.int64), nb_tiles_per_gaussian)

    nb_tiles_u = (W + T - 1) // T
    flat_tile_id = flat_tile_v * nb_tiles_u + flat_tile_u

    idx_z_order = torch.arange(nb_gaussians, device=pos.device, dtype=torch.int64)
    M = nb_gaussians + 1
    comp = flat_tile_id * M + idx_z_order[gaussian_ids]
    comp_sorted, perm = torch.sort(comp)
    gaussian_ids = gaussian_ids[perm]
    tile_ids_1d = torch.div(comp_sorted, M, rounding_mode="floor")

    unique_tile_ids, nb_gaussian_per_tile = torch.unique_consecutive(tile_ids_1d, return_counts=True)
    start = torch.zeros_like(unique_tile_ids)
    start[1:] = torch.cumsum(nb_gaussian_per_tile[:-1], dim=0)
    end = start + nb_gaussian_per_tile

    inverse_covariance = inv2x2(sigma_camera)
    inverse_covariance[:, 0, 0] = torch.clamp(inverse_covariance[:, 0, 0], min=min_conis)
    inverse_covariance[:, 1, 1] = torch.clamp(inverse_covariance[:, 1, 1], min=min_conis)

    final_image = torch.zeros((H * W, 3), device=pos.device, dtype=pos.dtype)

    for tile_id, s0, s1 in zip(unique_tile_ids.tolist(), start.tolist(), end.tolist()):
        current_gaussian_ids = gaussian_ids[s0:s1]

        txi = tile_id % nb_tiles_u
        tyi = tile_id // nb_tiles_u
        x0, y0 = txi * T, tyi * T
        x1, y1 = min((txi + 1) * T, W), min((tyi + 1) * T, H)
        if x0 >= x1 or y0 >= y1:
            continue

        xs = torch.arange(x0, x1, device=pos.device, dtype=pos.dtype)
        ys = torch.arange(y0, y1, device=pos.device, dtype=pos.dtype)
        pu, pv = torch.meshgrid(xs, ys, indexing="xy")
        px_u = pu.reshape(-1)
        px_v = pv.reshape(-1)
        pixel_idx_1d = (px_v * W + px_u).to(torch.int64)

        gaussian_i_u = u[current_gaussian_ids]
        gaussian_i_v = v[current_gaussian_ids]
        gaussian_i_color = color[current_gaussian_ids]
        gaussian_i_opacity = opacity[current_gaussian_ids]
        gaussian_i_inverse_covariance = inverse_covariance[current_gaussian_ids]

        du = px_u.unsqueeze(0) - gaussian_i_u.unsqueeze(-1)
        dv = px_v.unsqueeze(0) - gaussian_i_v.unsqueeze(-1)

        A11 = gaussian_i_inverse_covariance[:, 0, 0].unsqueeze(-1)
        A12 = gaussian_i_inverse_covariance[:, 0, 1].unsqueeze(-1)
        A22 = gaussian_i_inverse_covariance[:, 1, 1].unsqueeze(-1)

        q = A11 * du * du + 2 * A12 * du * dv + A22 * dv * dv
        inside = q <= chi_square_clip
        g = torch.exp(-0.5 * torch.clamp(q, max=chi_square_clip))
        g = torch.where(inside, g, torch.zeros_like(g))

        alpha_i = (gaussian_i_opacity.unsqueeze(-1) * g).clamp_max(alpha_max)
        alpha_i = torch.where(alpha_i >= alpha_cutoff, alpha_i, torch.zeros_like(alpha_i))

        one_minus_alpha_i = 1 - alpha_i
        T_i = torch.cumprod(one_minus_alpha_i, dim=0)
        T_i = torch.cat([torch.ones((1, alpha_i.shape[-1]), device=pos.device, dtype=pos.dtype), T_i[:-1]], dim=0)
        alive = (T_i > 1e-4).float()
        w = alpha_i * T_i * alive

        T_final = torch.prod(one_minus_alpha_i, dim=0)  # [T * T]
        tile_color = (w.unsqueeze(-1) * gaussian_i_color.unsqueeze(1)).sum(dim=0)
        tile_color = tile_color + T_final.unsqueeze(-1) * bg.unsqueeze(0)

        final_image[pixel_idx_1d] = tile_color

    return final_image.reshape((H, W, 3)).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Gaussian renderer")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Execution device")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of rendered frames")
    parser.add_argument("--output-dir", default="novel_views", help="Base output folder for rendered frames (saved in a scene subfolder)")
    parser.add_argument("--antialiasing", dest="antialiasing", action="store_true", help="Enable antialiasing (default)")
    parser.add_argument("--no-antialiasing", dest="antialiasing", action="store_false", help="Disable antialiasing")
    parser.set_defaults(antialiasing=True)
    parser.add_argument("--resolution-divisor", type=int, default=2, help="Render at 1/divisor of source resolution")

    parser.add_argument("--model-path", default=None, help="Path to trained model folder (contains point_cloud/iteration_*/point_cloud.ply)")
    parser.add_argument("--source-path", default=None, help="Path to source scene folder (contains sparse/0 for COLMAP)")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load for original format (-1 = latest)")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Which camera split to render in original format")
    parser.add_argument("--images", default=None, help="Images folder name inside source-path (default: images)")
    parser.add_argument("--depths", default="", help="Optional depths folder name inside source-path")
    parser.add_argument("--eval", action="store_true", help="Enable train/test split logic for COLMAP scenes")
    parser.add_argument("--trajectory-file", default=None, help="Optional file with custom c2w poses [N,4,4] for original mode")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")

    device = torch.device(args.device)
    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_path is None or args.source_path is None:
        raise ValueError("--model-path and --source-path are required")

    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.graphics_utils import fov2focal
    from utils.system_utils import searchForMaxIteration

    point_cloud_root = Path(args.model_path) / "point_cloud"
    iteration = searchForMaxIteration(str(point_cloud_root)) if args.iteration == -1 else args.iteration
    ply_path = point_cloud_root / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Could not find Gaussian PLY at {ply_path}")

    pos, opacity_raw, f_dc, f_rest, scale_raw, q_raw = load_gaussians_from_ply(str(ply_path), device)
    sigma = build_sigma_from_params(scale_raw, q_raw)

    scene_info = sceneLoadTypeCallbacks["Colmap"](
        args.source_path,
        args.images,
        args.depths,
        args.eval,
        train_test_exp=False,
    )

    cam_infos = scene_info.train_cameras if args.split == "train" else scene_info.test_cameras
    if len(cam_infos) == 0:
        raise RuntimeError(f"No cameras found for split '{args.split}'. Try changing --split/--eval.")

    if args.trajectory_file:
        orbit_c2ws = torch.load(args.trajectory_file, map_location=device)
        if orbit_c2ws.ndim != 3 or orbit_c2ws.shape[-2:] != (4, 4):
            raise ValueError(f"Trajectory file must contain poses with shape [N,4,4], got {tuple(orbit_c2ws.shape)}")

        frame_iter = orbit_c2ws if args.max_frames is None else orbit_c2ws[: args.max_frames]
        ref_cam = cam_infos[0]
        H_src, W_src = ref_cam.height, ref_cam.width
        H = H_src // args.resolution_divisor
        W = W_src // args.resolution_divisor
        fx = fov2focal(ref_cam.FovX, ref_cam.width)
        fy = fov2focal(ref_cam.FovY, ref_cam.height)
        cx, cy = W_src / 2.0, H_src / 2.0
        fx, fy, cx, cy = scale_intrinsics(H, W, H_src, W_src, fx, fy, cx, cy)

        for i, c2w_i in tqdm(enumerate(frame_iter), total=None if args.max_frames is None else len(frame_iter)):
            c2w = c2w_i.to(device=device, dtype=pos.dtype)
            color = evaluate_sh(f_dc, f_rest, pos, c2w)
            img = render(
                pos,
                color,
                opacity_raw,
                sigma,
                c2w,
                H,
                W,
                fx,
                fy,
                cx,
                cy,
                antialiasing=args.antialiasing,
            )
            Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(out_dir / f"frame_{i:05d}.png")
        return

    if args.max_frames is not None:
        cam_infos = cam_infos[: args.max_frames]

    for i, cam in tqdm(enumerate(cam_infos), total=len(cam_infos)):
        c2w = camera_info_to_c2w(cam, device)
        H_src, W_src = cam.height, cam.width
        H = H_src // args.resolution_divisor
        W = W_src // args.resolution_divisor
        fx = fov2focal(cam.FovX, cam.width)
        fy = fov2focal(cam.FovY, cam.height)
        cx, cy = W_src / 2.0, H_src / 2.0
        fx, fy, cx, cy = scale_intrinsics(H, W, H_src, W_src, fx, fy, cx, cy)

        color = evaluate_sh(f_dc, f_rest, pos, c2w)
        img = render(
            pos,
            color,
            opacity_raw,
            sigma,
            c2w,
            H,
            W,
            fx,
            fy,
            cx,
            cy,
            antialiasing=args.antialiasing,
        )
        Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(out_dir / f"frame_{i:05d}.png")


if __name__ == "__main__":
    main()
