#!/usr/bin/env python3
"""deploy_stent.py (batch‑aware edition)
---------------------------------------------------------------
Command‑line tool that deploys a crimped stent inside a vascular
surface mesh until the stent radius reaches – but never exceeds –
a prescribed target radius.

The script wraps the Kelvinlet‑based *one‑step* deformation routine
`deform_mesh_sdf_contact(...)` you supplied.  We keep calling that
routine, accumulating the increment each time, **but we stop the
loop the moment adding the next increment would overshoot the target
radius.**  (In that case the last deformation is **not** applied –
this exactly matches your requirement "stop once this would overshoot".)

Deploy one **or many** crimped stents inside a vascular surface mesh using
Kelvinlet SDF‑contact pushes.  For each stent we keep incrementing the stent
radius until the next increment would overshoot the prescribed *target*
radius – then we stop **without** applying that last step.

Single‑stent usage (identical to previous version)
-----
python deploy_stent.py  \
       --mesh   input_surface_mesh.vtp  \
       --cline  input_centerline.vtp  \
       --start   123                  # centre‑line point id of the distal tip of stent

optional flags  (see `-h` for the full list):
  --target‑R    0.40   # [cm] desired deployed stent radius
  --start‑R     0.05   # [cm] crimped radius (defaults to 0.05)
  --length      3.0    # [cm] stent length along centre‑line
  --out‑mesh    deployed_surface.vtp
  --out‑cl      deployed_centerline.vtp

Example:
python deploy_stent.py --mesh surf.vtp --cline cl.vtp \
                       --start 980 --length 3.0 --target-R 0.8

Batch mode
----------
python deploy_stent.py --mesh surf.vtp --cline cl.vtp --batch stents.txt

`stents.txt` (whitespace **or** comma separated), one line per stent::

    # startId   length(cm)   targetR(cm)   [startR(cm)]
    980         2            0.80          0.05
    1023        3            0.60          0.05
    745         1.5          0.75          0.04

Options
-------
--out-mesh  final_surface.vtp
--out-cl    final_centerline.vtp
"""
# ---------------------------------------------------------------------------
import time
import_start_time = time.time()
import argparse, pathlib, re
from collections import defaultdict

from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter
# from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtkmodules.vtkCommonCore import vtkIdTypeArray, vtkLongArray

import numpy as np
import jax.numpy as jnp
import jax as jx
from scipy.spatial import cKDTree
print(f"Total import time: {time.time() - import_start_time:.4f} seconds")

# a manual copy of the vtk_to_numpy from vtk.util.numpy_support codebase to speed up import
# -----------------------------------------------------------------------------
def get_vtk_to_numpy_typemap():
    """Returns the VTK array type to numpy array type mapping."""
    VTK_VOID            = 0
    VTK_BIT             = 1
    VTK_CHAR            = 2
    VTK_SIGNED_CHAR     =15
    VTK_UNSIGNED_CHAR   = 3
    VTK_SHORT           = 4
    VTK_UNSIGNED_SHORT  = 5
    VTK_INT             = 6
    VTK_UNSIGNED_INT    = 7
    VTK_LONG            = 8
    VTK_LONG_LONG       = 16
    VTK_UNSIGNED_LONG   = 9
    VTK_UNSIGNED_LONG_LONG = 17
    VTK_FLOAT           =10
    VTK_DOUBLE          =11
    VTK_ID_TYPE         =12
    VTK_ID_TYPE_SIZE = vtkIdTypeArray().GetDataTypeSize()
    if VTK_ID_TYPE_SIZE == 4:
        ID_TYPE_CODE = np.int32
    elif VTK_ID_TYPE_SIZE == 8:
        ID_TYPE_CODE = np.int64
    VTK_LONG_TYPE_SIZE = vtkLongArray().GetDataTypeSize()
    if VTK_LONG_TYPE_SIZE == 4:
        LONG_TYPE_CODE = np.int32
        ULONG_TYPE_CODE = np.uint32
    elif VTK_LONG_TYPE_SIZE == 8:
        LONG_TYPE_CODE = np.int64
        ULONG_TYPE_CODE = np.uint64

    _vtk_np = {VTK_BIT:np.uint8,
                VTK_CHAR:np.int8,
                VTK_SIGNED_CHAR:np.int8,
                VTK_UNSIGNED_CHAR:np.uint8,
                VTK_SHORT:np.int16,
                VTK_UNSIGNED_SHORT:np.uint16,
                VTK_INT:np.int32,
                VTK_UNSIGNED_INT:np.uint32,
                VTK_LONG:LONG_TYPE_CODE,
                VTK_LONG_LONG:np.int64,
                VTK_UNSIGNED_LONG:ULONG_TYPE_CODE,
                VTK_UNSIGNED_LONG_LONG:np.uint64,
                VTK_ID_TYPE:ID_TYPE_CODE,
                VTK_FLOAT:np.float32,
                VTK_DOUBLE:np.float64}
    return _vtk_np

def get_numpy_array_type(vtk_array_type):
    """Returns a numpy array typecode given a VTK array type."""
    return get_vtk_to_numpy_typemap()[vtk_array_type]

def v2n(vtk_array):
    """Converts a VTK data array to a numpy array.
    Given a subclass of vtkDataArray, this function returns an
    appropriate numpy array containing the same data -- it actually
    points to the same data.
    Parameters
    vtk_array
      The VTK data array to be converted.
    """
    typ = vtk_array.GetDataType()
    assert typ in get_vtk_to_numpy_typemap().keys(), \
           "Unsupported array type %s"%typ

    shape = vtk_array.GetNumberOfTuples(), \
            vtk_array.GetNumberOfComponents()

    # Get the data via the buffer interface
    dtype = get_numpy_array_type(typ)
    try:
        if typ != 1:
            result = np.frombuffer(vtk_array, dtype=dtype)
        else:
            result = np.unpackbits(vtk_array, count=shape[0])
    except ValueError:
        # http://mail.scipy.org/pipermail/numpy-tickets/2011-August/005859.html
        # numpy 1.5.1 (and maybe earlier) has a bug where if frombuffer is
        # called with an empty buffer, it throws ValueError exception. This
        # handles that issue.
        if shape[0] == 0:
            # create an empty array with the given shape.
            result = np.empty(shape, dtype=dtype)
        else:
            raise
    if shape[1] == 1:
        shape = (shape[0], )
    try:
        result.shape = shape
    except ValueError:
        if shape[0] == 0:
           # Refer to https://github.com/numpy/numpy/issues/2536 .
           # For empty array, reshape fails. Create the empty array explicitly
           # if that happens.
           result = np.empty(shape, dtype=dtype)
        else: raise
    return result

# -----------------------------------------------------------------------------
# I/O helpers -------------------------------------------------------------
def read_vtp(fname:str):
    r = vtkXMLPolyDataReader()
    r.SetFileName(fname)
    r.Update()
    return r.GetOutput()

def write_vtp(poly, fname:str):
    w = vtkXMLPolyDataWriter() 
    w.SetFileName(fname) 
    w.SetInputData(poly) 
    w.Write()

# --------------------------- scaling / sdf deform imports -------------------
def get_a_b(mu, nu):
    a = 1 / (4 * jnp.pi * mu)
    b = a / (4 * (1 - nu))
    return a, b

def mix(a, b, t): # = a * (1 - t) + b * t
    return a + (b - a) * t 

def smin_and_gradient(a, da, b, db, k=0.01):
    k = k * 4.0
    h = jnp.maximum(k - jnp.abs(a - b), 0.0) / k
    n = 0.5 * h
    m = h**2 * k / 4.0
    # Use jnp.where to choose between the two cases in a jittable way
    value = jnp.where(a < b, a - m, b - m)
    grad  = jnp.where(a < b, mix(da, db, n), mix(da, db, 1.0 - n))
    return value, grad

def fold_smin(carry, elem):
    cur_min_d, cur_min_dir = carry 
    d, dir = elem      # new distance and direction to combine
    new_d, new_dir = smin_and_gradient(cur_min_d, cur_min_dir, d, dir)
    return (new_d, new_dir), None

def compute_min_dist_and_direction(d, dir):
    # d: (num_segments,), dir: (num_segments, ndims)
    (final_d, final_dir), _ = jx.lax.scan(fold_smin, (d[0], dir[0]), (d[1:], dir[1:]))
    return final_d, final_dir

def force_kernel(a, b, eps, r):
    r_eps = (r**2 + eps**2)**0.5
    return (a-b)/r_eps + b/r_eps**3 + a/2*eps**2/r_eps**3

@jx.jit
def stent_bounding_box(data_points, stent_vertices, target_stent_radius, doi, doc):
    # Compute the minimum and maximum coordinates of the bounding box
    min_coords = jnp.min(stent_vertices, axis=0) - target_stent_radius - doi - doc - 0.01
    max_coords = jnp.max(stent_vertices, axis=0) + target_stent_radius + doi + doc + 0.01
    mask = jnp.all((data_points >= min_coords) & (data_points <= max_coords), axis=1)
    return mask

@jx.jit
def smin_sdf_capsule_contact_sculp(rv, a, b, stent_vertices, eps, s, r_target, r_current):
    doi = 0.65 # distance of influence: width of the deformation zone
    doc = 0.01 # distance within which contact is made
    # doi = 0.15
    # f_scale = 0.25 * doi * 0.1
    # f_scale = 0.25 * doi
    # rv = jnp.array(rv)
    ba_all = jnp.diff(stent_vertices, axis=0)
    pa_all = rv - stent_vertices[None, :-1, :]
    ba_dot_pa_all = jnp.sum(pa_all * ba_all[None, :, :], axis=-1)
    ba_dot_ba_all = jnp.sum(ba_all**2, axis=-1)
    h_all = jnp.clip(ba_dot_pa_all / ba_dot_ba_all, 0, 1)
    axis_to_point_all = pa_all - h_all[:, :, None] * ba_all[None, :, :]
    dist_all = jnp.linalg.norm(axis_to_point_all, axis=-1)[..., None]
    direction_all = axis_to_point_all / dist_all
    dist_all_squeezed = jnp.squeeze(dist_all, axis=-1)  # shape: (num_mesh_points, num_segments) 
    dist_to_surface_all = dist_all_squeezed - r_current
    # Vectorize the folding over all mesh points:
    final_dist_to_surface, final_direction = jx.vmap(compute_min_dist_and_direction)(dist_to_surface_all, direction_all)
    final_dist_to_surface = final_dist_to_surface[:, None]
    
    return final_dist_to_surface, final_direction

def get_sdf_contact_displacements(data, a, b, stent_vertices, eps, s, surface_mesh_scale_factor, force_center_normal, stent_halflength, target_stent_radius, current_stent_radius):
    force_center_point_id = data["nodes"]["force_center_point_id"]
    print("force center: ", force_center_point_id)
    data_points = data["points"]["surface"]
    # centerline_points = data["points"]["centerline"]
    num_kelvinlet_points = 1
    doi = 0.65
    doc = 0.01
    f_scale = 0.25 * doi * 0.1
    start_time = time.time()
    bb_mask = stent_bounding_box(data_points, stent_vertices, target_stent_radius, doi, doc)
    print("time taken to compute bounding box: ", time.time() - start_time)
    start_time = time.time()
    bb_mask = np.array(bb_mask)
    data_points = np.array(data_points) 
    data_points_masked = data_points[bb_mask]
    print("time taken to cast to numpy and mask out bounding box: ", time.time() - start_time)

    xs_np = np.expand_dims(data_points_masked, 1)
    xs = np.tile(xs_np, (1, num_kelvinlet_points, 1))
    # xs = jnp.array(xs)
    # xs = jnp.expand_dims(data_points, 1)
    # xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
   
    num_mesh_points = data_points.shape[0]
    start_time = time.time()
    total_num_vertices = xs.shape[0]
    print("total_num_vertices: ", total_num_vertices)
    final_dist_to_surface, final_direction = smin_sdf_capsule_contact_sculp(xs, a, b, stent_vertices, eps, s, target_stent_radius, current_stent_radius) #JIT-compiled
    new_contact_mask = (final_dist_to_surface < doc).astype(bool)
    # print("new_contact_mask shape: ", new_contact_mask.shape)
    print("time taken to compute new_contact points: ", time.time() - start_time)
    # xs = np.array(xs)
    xs = xs_np
    new_contact_mask = np.array(new_contact_mask)
    start_time = time.time()
    in_contact_vertices = xs[new_contact_mask, :]
    print("in_contact_vertices type: ", type(in_contact_vertices))
    print("time taken to obtain in_contact vertices subslice: ", time.time() - start_time)
    if in_contact_vertices.shape[0] == 0: # things are in contact <=> things are in influence
        step_size = f_scale * (-s)
        displacements, step_size = np.zeros((num_mesh_points, 3)), step_size
        return displacements, step_size 
    
    start_time = time.time()
    contact_tree = cKDTree(in_contact_vertices, leafsize=32)
    dist_min, _ = contact_tree.query(xs[:, 0, :], k=1, distance_upper_bound=doi)
    print("dist_min shape: ", dist_min.shape)
    print("time for KD Tree query: ", time.time() - start_time)
    start_time = time.time()
    in_influence_mask = dist_min < doi
    # print("in_influence_mask shape: ", in_influence_mask.shape)
    in_influence_indices = np.flatnonzero(in_influence_mask)
    print("in_influence_indices shape: ", in_influence_indices.shape)
    print("time taken to flattennonzero: ", time.time() - start_time)
    
    start_time = time.time()
    in_influence_to_in_contact_distances = dist_min[in_influence_mask]
    print("time taken to obtain in-influence vertices: ", time.time() - start_time)
    part_two_start_time = time.time()
    
    print("time for converting to jnp arrays: ", time.time() - part_two_start_time)
    start_time = time.time()
    doi_mask = (final_dist_to_surface < doi).astype(int)
    in_influence_vertices_blended_alpha_mask = np.zeros(total_num_vertices)
    # in_influence_vertices_blended_alpha = smin_sdf_capsule_contact_sculp_part_two_KD(in_influence_vertices, in_contact_vertices, doi)
    in_influence_vertices_blended_alpha = (1 - in_influence_to_in_contact_distances / doi)
    print("time taken to compute JIT sculpt part two: ", time.time() - start_time)
    start_time = time.time()
    in_influence_vertices_blended_alpha = np.array(in_influence_vertices_blended_alpha)
    in_influence_vertices_blended_alpha_mask[in_influence_indices] = in_influence_vertices_blended_alpha
    print("time taken to compute blended alpha mask in np: ", time.time() - start_time)
    start_time = time.time()
    displacements = f_scale * ((final_dist_to_surface / doi) ** 2 - 1) ** 2 * (-s) * final_direction * doi_mask * in_influence_vertices_blended_alpha_mask[:, None]
    full_displacements = np.zeros((num_mesh_points, 3))
    full_displacements[bb_mask] = displacements
    # displacements = f_scale * force_kernel(1.99, 0.88, 2.3, final_dist_to_surface) * (-s) * final_direction * doi_mask * in_influence_vertices_blended_alpha_mask[:, None]
    step_size = f_scale * (-s)
    print("time taken to compute rest of the displacements: ", time.time() - start_time)   
    # displacement is of type jnp array

    return full_displacements, step_size

# helpers from vtk_utils
# -----------------------------------------------------------------------------
def polydata_to_np_jnp_data(surface_polydata, centerline_polydata):
    # Convert to JAX-compatible arrays by using jnp.array
    surface_points_view_np = v2n(surface_polydata.GetPoints().GetData())
    centerline_points_view_np = v2n(centerline_polydata.GetPoints().GetData())
    # surface_points_jnp = surface_points_view_np
    # centerline_points_jnp = centerline_points_view_np
    # Create a dictionary to store data, including the JAX arrays
    data = {
        "points": {
            "surface_points_view_np": surface_points_view_np,
            "centerline_points_view_np": centerline_points_view_np,
            "surface": surface_points_view_np,
            "centerline": centerline_points_view_np
        },
        "nodes": {
            "all_indices": [],
            "force_center_point_id": -1
        },
        "centerline_coordinate": jnp.array([])
    }
    
    return data

def polydata_to_parent_tip_map(centerline_polydata):
    """
    Build a mapping  pointId -> maxPointId_of_closest_parent_segment
    for a VTK centre‑line tree whose segments are encoded by a
    {0,1}-flag array (one component per leaf branch).
    Returns dict { pointId (int) : parent_tip_pointId (int) }.
    """
    vtk_arr = centerline_polydata.GetPointData().GetArray("CenterlineId")
    if vtk_arr is None:
        raise ValueError("Point array 'CenterlineId' not found.")
    flags = v2n(vtk_arr) # (N, n_components)
    # print("flags.shape = ", flags.shape)
    # Group points into segments
    # flags: (n_pts, n_comp)
    # unique_rows: (n_segments, n_components)
    # inverse: (N,)   i --> segment_id for point i
    unique_rows, inverse = np.unique(flags, axis=0, return_inverse=True)
    segment_points = defaultdict(list)  # seg_id -> [pt_id, ...]
    for pointId, seg_id in enumerate(inverse):
        segment_points[seg_id].append(pointId)
    #  Pre‑compute the “tip” (largest point id) of every segment.
    seg_tip = {seg_id: max(pts) for seg_id, pts in segment_points.items()}
    segment_base_mask = np.zeros(flags.shape[0], dtype=bool)
    for seg_id, pts in segment_points.items():
        segment_base_mask[min(pts)] = True
    #  Pre‑compute bit counts (how many 1’s) to choose closest parent.
    seg_bitcount = unique_rows.sum(axis=1)      # (n_segments,)
    # 3)  For every segment, find its closest ancestor
    #     (superset with minimal extra 1‑bits)
    parent_tip_for_segment = {}   # seg_id -> parent_tip_point_id
    for child_id, child_mask in enumerate(unique_rows):
        # Vectorised superset test:
        # parent is superset  <=>   all 1‑bits in child also 1 in parent
        mask_ok = np.logical_or(~child_mask.astype(bool), unique_rows.astype(bool))
        is_superset = mask_ok.all(axis=1)
        # Exclude itself keep only strictly larger superset bit masks
        is_superset[child_id] = False
        # If no ancestor exists (root), map to its own tip.
        if not np.any(is_superset):
            parent_tip_for_segment[child_id] = seg_tip[child_id]
            continue
        # Among supersets pick the one with the fewest 1‑bits
        candidate_ids = np.nonzero(is_superset)[0]
        extra_bits = seg_bitcount[candidate_ids] - seg_bitcount[child_id]
        best_parent_idx = candidate_ids[np.argmin(extra_bits)]
        parent_tip_for_segment[child_id] = seg_tip[best_parent_idx]

    # 4)  Build the final point‑level dictionary
    point_to_parent_tip = {}
    for pt_id, seg_id in enumerate(inverse):
        point_to_parent_tip[pt_id] = parent_tip_for_segment[seg_id]
    # print("unit test: parent id for 662 = ", point_to_parent_tip[662])
    # print("unit test: parent id for 7745 = ", point_to_parent_tip[7745])
    return point_to_parent_tip, segment_base_mask

def sample_stent_axis_vertices(points, parent_tip_map, segment_base_mask, starting_point_idx, desired_total_length, desired_segment_length, sampling_direction=-1):
    """
    Extracts and resamples a subsegment of a polyline.
    
    Parameters:
      points (np.array): Nx3 array of 3D coordinates representing the polyline.
      desired_segment_length (float): The spacing (in cm) between resampled points.
      starting_point_idx (int): Index in points where the subsegment starts.
      desired_total_length (float): The desired total arc length (in cm) for the subsegment.
      jump_threshold (float): Heuristic length for detecting a jump (default: distance more than 1.0 cm is considered a jump).
    
    Returns:
      jax.numpy.array: A new array of 3D coordinates representing the resampled subsegment.
    
    The function iterates from the starting index, accumulating arc length. The sampling direction can be either -1 (backward) or +1 (forward), this ensures we always sample from the distal part of the branch towards a trunk so that stent location can be unambiguously identified with just the starting point index.
    """
    if sampling_direction not in (-1, 1):
        raise ValueError("sampling_direction must be integer -1 or +1")
    if len(points) < 2:
        raise ValueError("Not enough points to form a polyline.")
    
    diffs_all = np.diff(points, axis=0)
    distances_all = np.linalg.norm(diffs_all, axis=1)
    
    subsegment_points = []
    subsegment_points.append(points[starting_point_idx])
    cumulative_length = 0.0
    n_points = len(points)
    
    # Walk along the polyline starting from starting_point_idx.
    i = starting_point_idx
    idx_end = 0 if sampling_direction == -1 else n_points - 1
    next_point_idx_offset = -1 if sampling_direction == -1 else 0
    while i != idx_end:
        # d = distances_all[i+next_point_dist_idx]
        # if d > jump_threshold:
            # Jump detected; break out without including the jump segment.
            # print(f"Jump detected at segment {i} -> {i+sampling_direction} (distance {d:.4f} cm).")
            # break
        if segment_base_mask[i]:
            next_i = parent_tip_map[i]
            d = np.linalg.norm(points[next_i] - points[i])
        else:
            next_i = i + sampling_direction
            d = distances_all[i + next_point_idx_offset]
        # If adding the full segment would exceed desired_total_length,
        # interpolate along this segment to hit the target exactly.
        if cumulative_length + d < desired_total_length:
            cumulative_length += d
            subsegment_points.append(points[next_i])
        else:
            remaining = desired_total_length - cumulative_length
            t = remaining / d 
            new_point = (1 - t) * points[i] + t * points[next_i]
            subsegment_points.append(new_point)
            cumulative_length += remaining
            break  # desired total length achieved, exit loop

        i = next_i
    # print(f"subsegment_points = {subsegment_points}")
    effective_total_length = cumulative_length
    if effective_total_length < desired_total_length:
        print(f"Subsegment truncated due to jump. Best achieved length = {effective_total_length:.4f} cm")
    
    subsegment_points = np.array(subsegment_points)
    # Now resample the subsegment to have points uniformly spaced by desired_segment_length.
    # Compute cumulative arc-length for the subsegment.
    diffs = np.diff(subsegment_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumu_length = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = cumu_length[-1]
    # Generate new arc-length values from 0 to total_length, with spacing desired_segment_length.
    new_s = np.arange(0, total_length, desired_segment_length)
    new_s = np.append(new_s, total_length)
    new_vertices = []
    j = 0  # current segment index in the subsegment
    for s in new_s:
        # Find the segment that contains arc-length s.
        while j < len(cumu_length) - 2 and cumu_length[j+1] < s:
            j += 1
        seg_delta = cumu_length[j+1] - cumu_length[j]
        t = 0 if seg_delta == 0 else (s - cumu_length[j]) / seg_delta
        interpolated_vertex = (1 - t) * subsegment_points[j] + t * subsegment_points[j+1]
        new_vertices.append(interpolated_vertex)
    
    return jnp.array(new_vertices)

# ----------------------------- CLI helpers ---------------------------------
def parse_batch_file(path, default_start_R):
    """Return list[dict] with keys start, length, target_R, start_R."""
    patt = re.compile(r'[,\s]+')
    stents = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith('#'):          # skip blank / comment
                continue
            cols = patt.split(ln)
            if len(cols) not in (3,4):
                raise ValueError(f"Bad line in {path}: “{ln}”")
            start,length,target = map(float, cols[:3])
            start = int(start)
            start_R = float(cols[3]) if len(cols)==4 else default_start_R
            stents.append(dict(start=start, length=length,
                               target_R=target, start_R=start_R))
    if not stents:
        raise ValueError("Batch file empty or no valid lines found.")
    return stents

# -------------------------- JAX data wrapper -------------------------------
def polydata_to_np_data(surface_pd, centerline_pd):
    surf = v2n(surface_pd.GetPoints().GetData())
    cl   = v2n(centerline_pd.GetPoints().GetData())
    return {"points":{"surface":surf, "centerline":cl},
            "nodes":{"all_indices":[], "force_center_point_id":-1}}

def define_nodes_affine(data, ids):  data["nodes"]["all_indices"]=jnp.array(ids)
def assign_force_location(data, pid): data["nodes"]["force_center_point_id"]=pid

# ------------------------------ One‑step push ------------------------------
def one_step(data,               
             stent_vertices,     # jnp (m,3) sampled axis
             cur_R:float,
             eps:float, force_scale:float, a:float, b:float,
             halflength:float, target_R:float):
    """Perform one SDF‑contact Kelvinlet push; return ΔR actually produced."""
    step_start_time = time.time()
    # call the routine that returns displacements + step size
    disp, dR = get_sdf_contact_displacements(
        data, a, b, stent_vertices, eps, force_scale,
        surface_mesh_scale_factor=None,
        force_center_normal=None,
        stent_halflength=halflength,
        target_stent_radius=target_R,
        current_stent_radius=cur_R)

    # if applying dR would overshoot -> tell caller and *do not* change mesh
    if cur_R + dR > target_R:
        print(f"Total step time: {time.time() - step_start_time:.4f} seconds")
        return 0.0  # target radius reached
    # otherwise apply displacement to surface mesh in‑place
    data["points"]["surface"] += disp
    print(f"Total step time: {time.time() - step_start_time:.4f} seconds")
    return dR
# ---------------------------------------------------------------------------
def deploy_one_stent(mesh_pd, cl_pd, data,
                     start_ids, length, start_R, target_R,
                     save_step, is_intermediate, mesh_prefix, cl_prefix,
                     a,b, eps, force_scale, halflength):
    """Deploy a single stent; return final radius."""
    # update node / force‑centre
    start_id = start_ids[-1]
    define_nodes_affine(data,[start_id])
    assign_force_location(data,start_id)

    # resample axis on **current** center‑line
    ptip_map, seg_base = polydata_to_parent_tip_map(cl_pd)
    axis = sample_stent_axis_vertices(data['points']['centerline'],
                                      ptip_map, seg_base,
                                      start_id, length, 0.1,
                                      sampling_direction=-1)
    cur_R = start_R
    # next_snap = start_R + save_step if save_step else None
    itr = 0
    while True:
        dR = one_step(data, axis, cur_R, eps, force_scale, a, b,
                      halflength, target_R)
        if dR<=0:          # would overshoot
            break
        cur_R += dR  
        itr += 1
        # -------- snapshot --------------------------------------------
        # if next_snap and cur_R >= next_snap - 1e-8 and cur_R < target_R:
        #   tag = f"_R{next_snap:.2f}".replace('.','p')
        #   next_snap += save_step
    suffix = "_".join(str(sid) for sid in start_ids)
    if is_intermediate:
        write_vtp(mesh_pd, f"{mesh_prefix}_{suffix}.vtp")
        write_vtp(cl_pd,   f"{cl_prefix}_{suffix}.vtp")
        
    return cur_R

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--mesh',   required=True)
    ap.add_argument('--cline',  required=True)
    #
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--batch',  help='text/CSV with many stents (see doc‑string)')
    g.add_argument('--start', type=int, help='single stent start id')
    #
    ap.add_argument('--length',   type=float, default=3.0)
    ap.add_argument('--target-R', type=float, default=0.4)
    ap.add_argument('--start-R',  type=float, default=0.05)
    #
    ap.add_argument('--save-step', type=float, default=0.0,
                    help='>0 → save intermediate snapshots every N cm')
    ap.add_argument('--out-mesh', default='deployed_surface.vtp')
    ap.add_argument('--out-cl',   default='deployed_centerline.vtp')
    args = ap.parse_args()

    mesh_pd = read_vtp(args.mesh)
    cl_pd   = read_vtp(args.cline)

    # build data dict once; surface points will be updated in‑place
    data = polydata_to_np_data(mesh_pd, cl_pd)

    mu, nu = 1.0, 0.2
    a, b = get_a_b(mu,nu)
    eps = 0.2
    force_scale = -1.0
    halflength = 0.2

    # ----------------------------- batch list ------------------------------
    if args.batch:
        stents = parse_batch_file(args.batch, args.start_R)
    else:  # single‑stent run
        stents=[dict(start=args.start, length=args.length,
                     target_R=args.target_R, start_R=args.start_R)]

    base_snap   = pathlib.Path(args.out_mesh).with_suffix('').name
    start_time  = time.time()

    snapshot_dir = pathlib.Path(f"{base_snap}_intermediate_stents")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    mesh_prefix = snapshot_dir / pathlib.Path(args.out_mesh).with_suffix('')
    cl_prefix   = snapshot_dir / pathlib.Path(args.out_cl).with_suffix('')

    n_stents = len(stents)
    start_ids = [st['start'] for st in stents]
    for k,st in enumerate(stents,1):
        print(f"\n==== Deploying stent {k}/{len(stents)}  "
              f"[start={st['start']}, L={st['length']}cm] ====")
        snap_prefix = f"{base_snap}_stent{k}"
        is_intermediate = (k < n_stents)
        fin_R = deploy_one_stent(mesh_pd, cl_pd, data,
                                 start_ids[:k], st['length'],
                                 st['start_R'], st['target_R'],
                                 args.save_step, is_intermediate, mesh_prefix, cl_prefix,
                                 a,b,eps,force_scale, halflength)
        print(f"  → final radius = {fin_R:.4f} cm")

    # --------------------------- write final -------------------------------
    suffix = "_".join(str(sid) for sid in start_ids)
    out_mesh = f"{args.out_mesh}".replace('.vtp','') + f"_{suffix}.vtp"
    out_cl   = f"{args.out_cl}".replace('.vtp','') + f"_{suffix}.vtp"
    write_vtp(mesh_pd, out_mesh)
    write_vtp(cl_pd, out_cl)
    print(f"\nAll done in {time.time()-start_time:.1f}s"
          f"\n   final surface  → {args.out_mesh}"
          f"\n   final center‑line → {args.out_cl}")

if __name__ == '__main__':
    main()