#!/usr/bin/env python3
"""
von Hamos 2D RXES Map Generator (with Curvature Correction & Energy Transfer)
===========================================================================

Creates a 2D incident-energy versus emission-energy or energy-transfer map from:

    1. One SPEC scan (with KohzuE and i2 columns)
    2. One Lambda TIFF image at each SPEC scan point
    3. Optional Reference TIFF image to calibrate and straighten the
       2D sagittal Bragg curvature ("smile") along the x-axis before
       summing over x.
"""

import os
import re
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LogNorm
from PIL import Image


# =============================================================================
# General helper functions
# =============================================================================

def natural_key(text):
    """Natural sorting key: image_2 sorts before image_10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def axis_edges(centers):
    """Convert a vector of pixel/axis centers into pixel edges for pcolormesh."""
    centers = np.asarray(centers, dtype=float)

    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])

    edges = np.empty(len(centers) + 1, dtype=float)

    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - centers[-2])

    return edges


# =============================================================================
# TIFF file handling
# =============================================================================

def find_tiff_files_for_scan(folder, spec_basename, scan_number):
    """Find TIFF files associated with one SPEC scan."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"TIFF folder does not exist:\n{folder}")

    simple_pattern = re.compile(
        rf"^{re.escape(spec_basename)}_{scan_number}_(\d+)\.(tif|tiff)$",
        re.IGNORECASE,
    )

    explicit_pattern = re.compile(
        rf"^{re.escape(spec_basename)}_scan{scan_number}_point(\d+)\.(tif|tiff)$",
        re.IGNORECASE,
    )

    files = []

    for filename in os.listdir(folder):
        simple_match = simple_pattern.match(filename)
        explicit_match = explicit_pattern.match(filename)

        if simple_match:
            image_number = int(simple_match.group(1))
            files.append((image_number, os.path.join(folder, filename)))
        elif explicit_match:
            image_number = int(explicit_match.group(1))
            files.append((image_number, os.path.join(folder, filename)))

    files.sort(key=lambda item: item[0])

    if not files:
        raise FileNotFoundError(
            f"No TIFF images found for SPEC scan #{scan_number}.\n\n"
            f"SPEC basename: {spec_basename}\n"
            f"TIFF folder: {folder}"
        )

    return [path for _, path in files]


def load_tiff(path):
    """Read one TIFF image as a 2D float64 NumPy array."""
    with Image.open(path) as image:
        data = np.asarray(image, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(
            f"TIFF image is not two dimensional:\n{os.path.basename(path)}"
        )

    return data


# =============================================================================
# SPEC file parsing
# =============================================================================

def parse_spec_scan(spec_filepath, scan_number):
    """Read one SPEC scan (KohzuE and i2 columns)."""
    if not os.path.isfile(spec_filepath):
        raise FileNotFoundError(f"SPEC file not found:\n{spec_filepath}")

    with open(spec_filepath, "r", errors="replace") as file:
        lines = file.readlines()

    scan_header_pattern = re.compile(r"^#S\s*(\d+)\b")
    scan_start = None

    for index, line in enumerate(lines):
        match = scan_header_pattern.match(line)
        if match and int(match.group(1)) == scan_number:
            scan_start = index
            break

    if scan_start is None:
        raise ValueError(f"SPEC scan #{scan_number} was not found.")

    scan_lines = []

    for index in range(scan_start, len(lines)):
        if index > scan_start and scan_header_pattern.match(lines[index]):
            break
        scan_lines.append(lines[index])

    xb_params = {}
    count_time_spec = None

    for line in scan_lines:
        if line.startswith("#XB "):
            parts = line[4:].split("=", 1)
            if len(parts) == 2:
                xb_params[parts[0].strip()] = parts[1].strip()
        elif line.startswith("#T "):
            fields = line[3:].split()
            if fields:
                try:
                    count_time_spec = float(fields[0])
                except ValueError:
                    pass

    header = {
        "scan_number": scan_number,
        "raw_S": scan_lines[0].strip(),
        "xb_params": xb_params,
        "count_time_spec": count_time_spec,
    }

    column_labels = None
    label_line_index = None

    for index, line in enumerate(scan_lines):
        if line.startswith("#L "):
            column_labels = line[3:].strip().split()
            label_line_index = index
            break

    if column_labels is None:
        raise ValueError(
            f"No #L data-column line was found in SPEC scan #{scan_number}."
        )

    lower_map = {
        label.lower(): (idx, label)
        for idx, label in enumerate(column_labels)
    }

    if "i2" not in lower_map:
        raise ValueError(
            f"SPEC scan #{scan_number} does not contain an 'i2' column."
        )

    if "kohzue" not in lower_map:
        raise ValueError(
            f"SPEC scan #{scan_number} does not contain a 'KohzuE' column."
        )

    energy_index, energy_column = lower_map["kohzue"]
    i2_index, _ = lower_map["i2"]

    energy_values = []
    i2_values = []

    for line in scan_lines[label_line_index + 1:]:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.split()

        if len(fields) <= max(energy_index, i2_index):
            continue

        try:
            energy_values.append(float(fields[energy_index]))
            i2_values.append(float(fields[i2_index]))
        except ValueError:
            continue

    if not energy_values:
        raise ValueError(f"No numerical scan-point data found in scan #{scan_number}.")

    header["energy_column"] = energy_column
    header["column_labels"] = column_labels

    return (
        np.asarray(energy_values, dtype=float),
        np.asarray(i2_values, dtype=float),
        header,
    )


# =============================================================================
# Flat-Crystal Sagittal Curvature Calibration & Shift-and-Sum
# =============================================================================

def calibrate_dispersion_curve(
    ref_image,
    roi_low,
    roi_high,
    bad_roi_low,
    bad_roi_high,
    fit_y_low,
    fit_y_high,
    x_bin_size=8,
    poly_degree=2,
    half_centroid_window=6,
):
    """Calibrate the iso-energy curvature y_peak(x) from a 2D reference image."""
    n_rows, n_cols = ref_image.shape

    if fit_y_low < 0 or fit_y_high >= n_rows or fit_y_low >= fit_y_high:
        raise ValueError(
            f"Fit y range [{fit_y_low}, {fit_y_high}] is invalid for image height {n_rows}."
        )

    valid_x = np.arange(roi_low, roi_high + 1, dtype=int)
    if bad_roi_low is not None and bad_roi_high is not None and bad_roi_low <= bad_roi_high:
        valid_x = valid_x[(valid_x < bad_roi_low) | (valid_x > bad_roi_high)]

    if valid_x.size < poly_degree + 2:
        raise ValueError("Not enough valid x columns to fit curvature.")

    x_centers = []
    y_peaks = []
    weights = []

    for start_idx in range(0, len(valid_x), max(1, int(x_bin_size))):
        chunk_cols = valid_x[start_idx:start_idx + max(1, int(x_bin_size))]
        if len(chunk_cols) == 0:
            continue

        profile = ref_image[fit_y_low:fit_y_high + 1, chunk_cols].sum(axis=1)
        if np.all(profile <= 0):
            continue

        local_max_idx = int(np.argmax(profile))
        peak_y_abs = fit_y_low + local_max_idx

        w_low = max(fit_y_low, peak_y_abs - half_centroid_window)
        w_high = min(fit_y_high, peak_y_abs + half_centroid_window)

        y_sub = np.arange(w_low, w_high + 1, dtype=float)
        I_sub = ref_image[w_low:w_high + 1, chunk_cols].sum(axis=1).astype(float)

        I_sub = I_sub - np.min(I_sub)
        total_weight = np.sum(I_sub)

        if total_weight > 0:
            y_centroid = np.sum(y_sub * I_sub) / total_weight
            x_centers.append(float(np.mean(chunk_cols)))
            y_peaks.append(float(y_centroid))
            weights.append(float(np.sqrt(total_weight)))

    x_centers = np.asarray(x_centers, dtype=float)
    y_peaks = np.asarray(y_peaks, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(x_centers) <= poly_degree:
        raise ValueError("Could not find enough valid emission peaks across x to fit polynomial.")

    poly_coeffs = np.polyfit(x_centers, y_peaks, deg=poly_degree, w=weights)
    residuals = y_peaks - np.polyval(poly_coeffs, x_centers)
    std_res = np.std(residuals)
    if std_res > 1e-6:
        inliers = np.abs(residuals) < 3.0 * std_res
        if np.sum(inliers) > poly_degree + 1:
            poly_coeffs = np.polyfit(
                x_centers[inliers],
                y_peaks[inliers],
                deg=poly_degree,
                w=weights[inliers],
            )

    x_ref = 0.5 * (roi_low + roi_high)
    y_ref = float(np.polyval(poly_coeffs, x_ref))

    all_x = np.arange(n_cols, dtype=float)
    dy_per_col = np.polyval(poly_coeffs, all_x) - y_ref

    return dy_per_col, poly_coeffs, x_centers, y_peaks, x_ref


def sum_columns_with_curvature_correction(image, valid_x, dy_per_col=None):
    """Sum `image` horizontally across `valid_x` columns with optional vertical shift."""
    if dy_per_col is None:
        return image[:, valid_x].sum(axis=1)

    n_rows = image.shape[0]
    y_grid = np.arange(n_rows, dtype=float)
    summed_spectrum = np.zeros(n_rows, dtype=np.float64)

    for col in valid_x:
        shift = dy_per_col[col]
        shifted_col = np.interp(
            y_grid + shift,
            y_grid,
            image[:, col],
            left=0.0,
            right=0.0,
        )
        summed_spectrum += shifted_col

    return summed_spectrum


# =============================================================================
# von Hamos emission-energy calibration
# =============================================================================

def compute_emission_energy(
    Eb_keV,
    Eref_keV,
    Pr_pixel,
    Ps_mm,
    RC_mm,
    pixels,
):
    """Calculate emission energy for every dispersive pixel along the y-axis."""
    if Eb_keV <= 0:
        raise ValueError("Eb must be greater than zero.")

    if Eref_keV <= 0:
        raise ValueError("Reference emission energy must be greater than zero.")

    if Eref_keV < Eb_keV:
        raise ValueError(
            "Reference emission energy Eref must be greater than or equal to Eb."
        )

    if Ps_mm <= 0:
        raise ValueError("Detector pixel size Ps must be greater than zero.")

    if RC_mm <= 0:
        raise ValueError("Source-to-analyzer distance RC must be greater than zero.")

    theta_ref_deg = np.degrees(np.arccos(Eb_keV / Eref_keV))
    RA_mm = RC_mm * np.cos(np.radians(theta_ref_deg))
    y_mm = (pixels - Pr_pixel) * Ps_mm
    theta_pixel_deg = theta_ref_deg + np.degrees(
        np.arctan((y_mm) / (2.0 * RA_mm))
    )

    cos_theta = np.cos(np.radians(theta_pixel_deg))

    if np.any(np.isclose(cos_theta, 0.0)):
        raise ValueError("Invalid geometry: cos(theta) is zero.")

    emission_energy_eV = (Eb_keV / cos_theta) * 1000.0
    return emission_energy_eV


def process_hamos_image(
    image,
    i2_value,
    mean_i2,
    dispersive_axis,
    roi_low,
    roi_high,
    bad_roi_low,
    bad_roi_high,
    Eb_keV,
    Eref_keV,
    Pr_pixel,
    Ps_mm,
    RC_mm,
    Ds_mm,
    count_time,
    dy_per_col=None,
):
    """Process one Lambda TIFF image for one KohzuE point."""
    if image.ndim != 2:
        raise ValueError("Detector image must be two dimensional.")

    if i2_value <= 0:
        raise ValueError(f"Invalid i2 value: {i2_value}")

    if count_time <= 0:
        raise ValueError("Counting time must be greater than zero.")

    if dispersive_axis == 1:
        n_rows_y = image.shape[0]
        n_cols_x = image.shape[1]

        if roi_low < 0 or roi_high >= n_cols_x:
            raise ValueError(
                f"Non-dispersive ROI x range [{roi_low}, {roi_high}] is outside image x range [0, {n_cols_x - 1}]."
            )

        valid_x = np.arange(roi_low, roi_high + 1, dtype=int)
        if bad_roi_low is not None and bad_roi_high is not None and bad_roi_low <= bad_roi_high:
            valid_x = valid_x[(valid_x < bad_roi_low) | (valid_x > bad_roi_high)]

        if valid_x.size == 0:
            raise ValueError("No valid non-dispersive x pixels remain after excluding bad pixels.")

        spectrum_full = sum_columns_with_curvature_correction(
            image=image,
            valid_x=valid_x,
            dy_per_col=dy_per_col,
        )

        if Ds_mm > 0:
            half_width_pixels = int(round(Ds_mm / Ps_mm))
            pmin = max(0, int(round(Pr_pixel)) - half_width_pixels)
            pmax = min(n_rows_y - 1, int(round(Pr_pixel)) + half_width_pixels)
        else:
            pmin = 0
            pmax = n_rows_y - 1

        pixels = np.arange(pmin, pmax + 1, dtype=float)
        raw_spectrum = spectrum_full[pmin:pmax + 1].astype(np.float64)

    elif dispersive_axis == 0:
        n_rows_y = image.shape[0]
        n_cols_x = image.shape[1]

        if roi_low < 0 or roi_high >= n_rows_y:
            raise ValueError(
                f"Non-dispersive ROI y range [{roi_low}, {roi_high}] is outside image y range [0, {n_rows_y - 1}]."
            )

        valid_y = np.arange(roi_low, roi_high + 1, dtype=int)
        if bad_roi_low is not None and bad_roi_high is not None and bad_roi_low <= bad_roi_high:
            valid_y = valid_y[(valid_y < bad_roi_low) | (valid_y > bad_roi_high)]

        if valid_y.size == 0:
            raise ValueError("No valid non-dispersive y pixels remain after excluding bad pixels.")

        spectrum_full = sum_columns_with_curvature_correction(
            image=image.T,
            valid_x=valid_y,
            dy_per_col=dy_per_col,
        )

        if Ds_mm > 0:
            half_width_pixels = int(round(Ds_mm / Ps_mm))
            pmin = max(0, int(round(Pr_pixel)) - half_width_pixels)
            pmax = min(n_cols_x - 1, int(round(Pr_pixel)) + half_width_pixels)
        else:
            pmin = 0
            pmax = n_cols_x - 1

        pixels = np.arange(pmin, pmax + 1, dtype=float)
        raw_spectrum = spectrum_full[pmin:pmax + 1].astype(np.float64)

    else:
        raise ValueError("Dispersive axis must be 1 (vertical y) or 0 (horizontal x).")

    if pmin > pmax:
        raise ValueError("Calculated dispersive pixel range is invalid.")

    normalized_spectrum = (raw_spectrum / i2_value / count_time) * mean_i2

    emission_energy_eV = compute_emission_energy(
        Eb_keV=Eb_keV,
        Eref_keV=Eref_keV,
        Pr_pixel=Pr_pixel,
        Ps_mm=Ps_mm,
        RC_mm=RC_mm,
        pixels=pixels,
    )

    return emission_energy_eV, normalized_spectrum, (pmin, pmax)


# =============================================================================
# GUI
# =============================================================================

class Hamos2DMapGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("von Hamos 2D RXES Map Generator (with Energy Transfer & Reversed Emission Axis)")
        self.root.geometry("1550x1010")
        self.root.minsize(1200, 840)

        self.spec_file_path = None
        self.spec_basename = None
        self.tiff_folder_path = None
        self.ref_tiff_path = None

        self.dy_per_col = None
        self.poly_coeffs = None
        self.x_ref_curve = None

        self.hamos_map = None
        self.incident_axis_eV = None
        self.emission_axis_eV = None
        self.i2_values = None
        self.metadata = None

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)

        self._build_controls()
        self._build_plot()

    def _build_controls(self):
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.grid(row=0, column=0, sticky="ns")

        ctrl.columnconfigure(0, weight=0)
        ctrl.columnconfigure(1, weight=1)

        self.entries = {}

        params = [
            ("Scan number", "23"),
            ("Eb (keV)", "9.685"),
            ("Reference pixel Pr (dispersive y)", "152"),
            ("Reference emission energy Eref (keV)", "9.685"),
            ("Pixel size Ps (mm/pixel)", "0.024013"),
            ("Source-analyzer distance RC (mm)", "2000.0"),
            ("Analyzer width Ds (mm, 0=full axis)", "1.3"),
            ("Non-dispersive ROI low pixel (x)", "50"),
            ("Non-dispersive ROI high pixel (x)", "450"),
            ("Exclude bad pixel low (x)", "254"),
            ("Exclude bad pixel high (x)", "261"),
            ("Dispersive axis: 1=y (vert), 0=x (horiz)", "1"),
            ("Counting time per image (s)", "30.0"),
            ("Curve fit y low pixel", "170"),
            ("Curve fit y high pixel", "235"),
            ("Curve fit x bin width (pixels)", "8"),
            ("Curve polynomial degree (1 or 2)", "2"),
        ]

        for row, (label, default) in enumerate(params):
            ttk.Label(ctrl, text=label).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=2,
            )

            entry = ttk.Entry(ctrl, width=18)
            entry.insert(0, default)
            entry.grid(
                row=row,
                column=1,
                sticky="ew",
                pady=2,
            )

            self.entries[label] = entry

        row = len(params)

        self.use_curve_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ctrl,
            text="Apply flat-crystal curvature correction",
            variable=self.use_curve_var,
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(4, 2),
        )

        row += 1

        ttk.Label(ctrl, text="Plot type").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )

        self.plot_type_var = tk.StringVar(value="Emission Energy")
        plot_type_cb = ttk.Combobox(
            ctrl,
            textvariable=self.plot_type_var,
            values=["Emission Energy", "Energy Transfer"],
            state="readonly",
            width=16,
        )
        plot_type_cb.grid(row=row, column=1, sticky="ew", pady=2)
        plot_type_cb.bind("<<ComboboxSelected>>", lambda e: self._replot())

        row += 1

        ttk.Label(ctrl, text="Incident energy unit").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )

        self.incident_unit_var = tk.StringVar(value="eV")

        ttk.Combobox(
            ctrl,
            textvariable=self.incident_unit_var,
            values=["keV", "eV"],
            state="readonly",
            width=16,
        ).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )

        row += 1

        ttk.Label(ctrl, text="Color scale").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )

        self.scale_var = tk.StringVar(value="linear")

        ttk.Combobox(
            ctrl,
            textvariable=self.scale_var,
            values=["linear", "log"],
            state="readonly",
            width=16,
        ).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )

        row += 1

        ttk.Label(ctrl, text="Colormap").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )

        self.cmap_var = tk.StringVar(value="viridis")

        ttk.Combobox(
            ctrl,
            textvariable=self.cmap_var,
            values=[
                "viridis",
                "inferno",
                "magma",
                "plasma",
                "hot",
                "jet",
                "Spectral",
            ],
            state="readonly",
            width=16,
        ).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )

        row += 1

        ttk.Label(ctrl, text="Color low / high percentile").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )

        pct_frame = ttk.Frame(ctrl)
        pct_frame.grid(row=row, column=1, sticky="ew", pady=2)
        pct_frame.columnconfigure(0, weight=1)
        pct_frame.columnconfigure(1, weight=1)

        self.pct_low_var = tk.DoubleVar(value=1.0)
        self.pct_high_var = tk.DoubleVar(value=99.5)

        ttk.Entry(pct_frame, textvariable=self.pct_low_var, width=8).grid(
            row=0, column=0, sticky="ew", padx=(0, 2)
        )
        ttk.Entry(pct_frame, textvariable=self.pct_high_var, width=8).grid(
            row=0, column=1, sticky="ew", padx=(2, 0)
        )

        row += 1

        self.spec_var = tk.StringVar(value="SPEC: not loaded")
        self.tiff_var = tk.StringVar(value="TIFF folder: not loaded")
        self.curve_var = tk.StringVar(value="Curvature calibration: none")

        ttk.Label(
            ctrl,
            textvariable=self.spec_var,
            foreground="darkgreen",
            wraplength=360,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 1))

        row += 1

        ttk.Label(
            ctrl,
            textvariable=self.tiff_var,
            foreground="blue",
            wraplength=360,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=1)

        row += 1

        ttk.Label(
            ctrl,
            textvariable=self.curve_var,
            foreground="purple",
            wraplength=360,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=1)

        row += 1

        button_frame = ttk.Frame(ctrl)
        button_frame.grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=4,
        )

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        ttk.Button(
            button_frame,
            text="1. Load SPEC File",
            command=self._load_spec,
        ).grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        ttk.Button(
            button_frame,
            text="2. Load TIFF Folder",
            command=self._load_tiff,
        ).grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        ttk.Button(
            button_frame,
            text="3. Load Reference TIFF & Fit Curve",
            command=self._calibrate_from_reference_tiff,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        ttk.Button(
            button_frame,
            text="Auto-Fill from SPEC #XB",
            command=self._autofill_from_spec,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        ttk.Button(
            button_frame,
            text="4. Generate von Hamos RXES Map",
            command=self._generate,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        ttk.Button(
            button_frame,
            text="Replot RXES Map",
            command=self._replot,
        ).grid(row=4, column=0, sticky="ew", padx=2, pady=2)

        ttk.Button(
            button_frame,
            text="Save Map Data",
            command=self._save_data,
        ).grid(row=4, column=1, sticky="ew", padx=2, pady=2)

        ttk.Button(
            button_frame,
            text="Save Figure",
            command=self._save_figure,
        ).grid(row=5, column=0, sticky="ew", padx=2, pady=2)

        ttk.Button(
            button_frame,
            text="TIFF Diagnostics",
            command=self._diagnose_tiffs,
        ).grid(row=5, column=1, sticky="ew", padx=2, pady=2)

        row += 1

        ttk.Label(ctrl, text="Log").grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(4, 2),
        )

        row += 1

        self.logbox = tk.Text(ctrl, width=48, height=10)
        self.logbox.grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="nsew",
        )

        scrollbar = ttk.Scrollbar(
            ctrl,
            orient="vertical",
            command=self.logbox.yview,
        )

        scrollbar.grid(row=row, column=2, sticky="ns")

        self.logbox.configure(yscrollcommand=scrollbar.set)
        ctrl.rowconfigure(row, weight=1)

    def _build_plot(self):
        plot_frame = ttk.Frame(self.root, padding=10)
        plot_frame.grid(row=0, column=1, sticky="nsew")

        plot_frame.rowconfigure(0, weight=1)
        plot_frame.rowconfigure(1, weight=0)
        plot_frame.columnconfigure(0, weight=1)

        canvas_frame = ttk.Frame(plot_frame)
        canvas_frame.grid(row=0, column=0, sticky="nsew")

        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")

        self.fig = plt.Figure(figsize=(10.5, 8.2))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("von Hamos 2D RXES Map")
        self.ax.set_xlabel("Incident Energy (eV)")
        self.ax.set_ylabel("Emission Energy (eV)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def log(self, message):
        self.logbox.insert(tk.END, str(message) + "\n")
        self.logbox.see(tk.END)
        self.root.update_idletasks()

    def set_entry(self, label, value):
        entry = self.entries[label]
        entry.delete(0, tk.END)
        entry.insert(0, str(value))

    def get_float(self, label):
        return float(self.entries[label].get())

    def get_int(self, label):
        return int(float(self.entries[label].get()))

    def get_optional_int(self, label):
        text = self.entries[label].get().strip()
        if text == "":
            return None
        return int(float(text))

    # -------------------------------------------------------------------------
    # Load SPEC, TIFF folder, and Reference TIFF for Curvature Calibration
    # -------------------------------------------------------------------------

    def _load_spec(self):
        path = filedialog.askopenfilename(
            title="Select SPEC Data File",
            filetypes=[
                ("SPEC data files", "*.dat *.spec *"),
                ("All files", "*"),
            ],
        )

        if not path:
            return

        self.spec_file_path = path
        self.spec_basename = os.path.splitext(os.path.basename(path))[0]

        self.spec_var.set(
            f"SPEC file: {os.path.basename(path)} (Basename: {self.spec_basename})"
        )

        self.log("=" * 68)
        self.log(f"SPEC file loaded: {path}")

        if self.tiff_folder_path is None:
            self.tiff_folder_path = os.path.dirname(path)
            self.tiff_var.set(f"TIFF folder: {self.tiff_folder_path}")

    def _load_tiff(self):
        initial_directory = (
            self.tiff_folder_path
            if self.tiff_folder_path
            else (os.path.dirname(self.spec_file_path) if self.spec_file_path else None)
        )

        folder = filedialog.askdirectory(
            title="Select Lambda TIFF Folder",
            initialdir=initial_directory,
        )

        if not folder:
            return

        self.tiff_folder_path = folder

        n_tiff = len([
            filename
            for filename in os.listdir(folder)
            if filename.lower().endswith((".tif", ".tiff"))
        ])

        self.tiff_var.set(f"TIFF folder: {folder} ({n_tiff} TIFFs)")
        self.log(f"TIFF folder loaded: {folder} ({n_tiff} files)")

    def _calibrate_from_reference_tiff(self):
        initial_dir = self.tiff_folder_path if self.tiff_folder_path else None
        path = filedialog.askopenfilename(
            title="Select Reference TIFF Image for Curvature Calibration",
            initialdir=initial_dir,
            filetypes=[
                ("TIFF images", "*.tif *.tiff"),
                ("All files", "*"),
            ],
        )

        if not path:
            return

        try:
            ref_image = load_tiff(path)

            roi_low = self.get_int("Non-dispersive ROI low pixel (x)")
            roi_high = self.get_int("Non-dispersive ROI high pixel (x)")
            bad_roi_low = self.get_optional_int("Exclude bad pixel low (x)")
            bad_roi_high = self.get_optional_int("Exclude bad pixel high (x)")

            fit_y_low = self.get_int("Curve fit y low pixel")
            fit_y_high = self.get_int("Curve fit y high pixel")
            x_bin_size = self.get_int("Curve fit x bin width (pixels)")
            poly_degree = self.get_int("Curve polynomial degree (1 or 2)")

            dy_per_col, poly_coeffs, x_centers, y_peaks, x_ref = calibrate_dispersion_curve(
                ref_image=ref_image,
                roi_low=roi_low,
                roi_high=roi_high,
                bad_roi_low=bad_roi_low,
                bad_roi_high=bad_roi_high,
                fit_y_low=fit_y_low,
                fit_y_high=fit_y_high,
                x_bin_size=x_bin_size,
                poly_degree=poly_degree,
            )

        except Exception as error:
            messagebox.showerror("Curvature Calibration Error", str(error))
            return

        self.ref_tiff_path = path
        self.dy_per_col = dy_per_col
        self.poly_coeffs = poly_coeffs
        self.x_ref_curve = x_ref
        self.use_curve_var.set(True)

        max_shift = np.max(np.abs(dy_per_col[roi_low:roi_high + 1]))
        self.curve_var.set(
            f"Calibrated: {os.path.basename(path)} (max |dy| = {max_shift:.2f} px)"
        )

        self.log("=" * 68)
        self.log(f"Curvature calibrated from: {os.path.basename(path)}")
        self.log(f"Polynomial coeffs (deg {poly_degree}): {np.array2string(poly_coeffs, precision=6)}")
        self.log(f"Reference x0 = {x_ref:.1f}, y(x0) = {np.polyval(poly_coeffs, x_ref):.2f} px")
        self.log(f"Max vertical shift across ROI [{roi_low}, {roi_high}]: {max_shift:.2f} pixels")

        valid_x = np.arange(roi_low, roi_high + 1, dtype=int)
        if bad_roi_low is not None and bad_roi_high is not None and bad_roi_low <= bad_roi_high:
            valid_x = valid_x[(valid_x < bad_roi_low) | (valid_x > bad_roi_high)]

        raw_1d = sum_columns_with_curvature_correction(ref_image, valid_x, dy_per_col=None)
        corr_1d = sum_columns_with_curvature_correction(ref_image, valid_x, dy_per_col=dy_per_col)

        self.fig.clear()
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        pos = ref_image[ref_image > 0]
        vmax = np.percentile(pos, 99.0) if pos.size > 0 else 1.0
        ax1.imshow(ref_image, cmap="gray", origin="upper", vmin=0, vmax=vmax, aspect="auto")

        x_plot = np.linspace(roi_low, roi_high, 200)
        y_plot = np.polyval(poly_coeffs, x_plot)

        ax1.plot(x_centers, y_peaks, "ro", markersize=4, label="Measured slice peaks")
        ax1.plot(x_plot, y_plot, "c-", linewidth=2, label="Fitted iso-energy curve")
        ax1.axhline(fit_y_low, color="yellow", linestyle="--", linewidth=0.8, label="Fit y window")
        ax1.axhline(fit_y_high, color="yellow", linestyle="--", linewidth=0.8)
        ax1.axvline(roi_low, color="lime", linestyle=":", linewidth=1.0)
        ax1.axvline(roi_high, color="lime", linestyle=":", linewidth=1.0)
        if bad_roi_low is not None and bad_roi_high is not None and bad_roi_low <= bad_roi_high:
            ax1.axvspan(bad_roi_low, bad_roi_high, color="red", alpha=0.25, label="Excluded bad x")

        ax1.set_title(f"Reference TIFF: {os.path.basename(path)}")
        ax1.set_xlabel("x pixel (non-dispersive)")
        ax1.set_ylabel("y pixel (dispersive)")
        ax1.legend(loc="lower right", fontsize=8)

        y_axis = np.arange(ref_image.shape[0])
        ax2.plot(y_axis, raw_1d, label="Uncorrected sum over x", color="tab:gray", alpha=0.8)
        ax2.plot(y_axis, corr_1d, label="Curvature-corrected sum", color="tab:red", linewidth=1.8)
        ax2.set_xlim(max(0, fit_y_low - 30), min(ref_image.shape[0] - 1, fit_y_high + 30))
        ax2.set_title("1D Emission Line (Before vs. After Straightening)")
        ax2.set_xlabel("Dispersive y pixel")
        ax2.set_ylabel("Summed Counts")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def _autofill_from_spec(self):
        if self.spec_file_path is None:
            messagebox.showwarning("No SPEC File", "Load a SPEC file first.")
            return

        try:
            scan_number = self.get_int("Scan number")
            _, _, header = parse_spec_scan(self.spec_file_path, scan_number)
        except Exception as error:
            messagebox.showerror("SPEC Read Error", str(error))
            return

        xb = header.get("xb_params", {})
        if "Analyzer_EB_keV" in xb:
            self.set_entry("Eb (keV)", xb["Analyzer_EB_keV"])
        if "Rowland_Radius_m" in xb:
            self.set_entry("Source-analyzer distance RC (mm)", xb["Rowland_Radius_m"])
        if "Analyzer_Crystal_Size_mm" in xb:
            self.set_entry("Analyzer width Ds (mm, 0=full axis)", xb["Analyzer_Crystal_Size_mm"])
        if "Lambda_Strip_Size_mm" in xb:
            self.set_entry("Pixel size Ps (mm/pixel)", xb["Lambda_Strip_Size_mm"])
        if header.get("count_time_spec") is not None:
            self.set_entry("Counting time per image (s)", header["count_time_spec"])

        self.log(f"Auto-filled geometry parameters from SPEC scan #{scan_number}.")

    def _diagnose_tiffs(self):
        if self.tiff_folder_path is None or self.spec_basename is None:
            messagebox.showwarning("Warning", "Load SPEC file and TIFF folder first.")
            return

        tiffs = sorted(
            [
                filename
                for filename in os.listdir(self.tiff_folder_path)
                if filename.lower().endswith((".tif", ".tiff"))
            ],
            key=natural_key,
        )

        self.log("=" * 68)
        self.log(f"Total TIFF files found: {len(tiffs)}")
        for filename in tiffs[:20]:
            self.log(f"  {filename}")
        self.log("=" * 68)

    # -------------------------------------------------------------------------
    # Generate 2D RXES Map
    # -------------------------------------------------------------------------

    def _generate(self):
        if self.spec_file_path is None or self.tiff_folder_path is None:
            messagebox.showwarning("Warning", "Load SPEC file and TIFF folder first.")
            return

        try:
            scan_number = self.get_int("Scan number")

            Eb_keV = self.get_float("Eb (keV)")
            Pr_pixel = self.get_float("Reference pixel Pr (dispersive y)")
            Eref_keV = self.get_float("Reference emission energy Eref (keV)")
            Ps_mm = self.get_float("Pixel size Ps (mm/pixel)")
            RC_mm = self.get_float("Source-analyzer distance RC (mm)")
            Ds_mm = self.get_float("Analyzer width Ds (mm, 0=full axis)")

            roi_low = self.get_int("Non-dispersive ROI low pixel (x)")
            roi_high = self.get_int("Non-dispersive ROI high pixel (x)")

            bad_roi_low = self.get_optional_int("Exclude bad pixel low (x)")
            bad_roi_high = self.get_optional_int("Exclude bad pixel high (x)")

            dispersive_axis = self.get_int("Dispersive axis: 1=y (vert), 0=x (horiz)")
            count_time = self.get_float("Counting time per image (s)")

            if roi_low > roi_high:
                raise ValueError("ROI low pixel must be <= ROI high pixel.")

        except ValueError as error:
            messagebox.showerror("Input Error", str(error))
            return

        apply_curve = self.use_curve_var.get() and (self.dy_per_col is not None)
        active_dy = self.dy_per_col if apply_curve else None

        self.log("=" * 68)
        self.log(f"Generating von Hamos RXES map for SPEC scan #{scan_number}")
        if bad_roi_low is not None and bad_roi_high is not None and bad_roi_low <= bad_roi_high:
            self.log(f"Excluding bad non-dispersive pixels: [{bad_roi_low}, {bad_roi_high}]")
        if apply_curve:
            self.log(
                f"Applying flat-crystal curvature correction from: "
                f"{os.path.basename(self.ref_tiff_path)}"
            )
        else:
            self.log("Curvature correction: OFF (straight summation)")

        try:
            incident_energy, i2_values, header = parse_spec_scan(
                self.spec_file_path,
                scan_number,
            )
        except Exception as error:
            messagebox.showerror("SPEC Read Error", str(error))
            return

        incident_unit = self.incident_unit_var.get().strip().lower()
        if incident_unit == "ev":
            incident_energy_eV = incident_energy * 1000.0
        else:
            incident_energy_eV = incident_energy.copy()

        try:
            tiff_files = find_tiff_files_for_scan(
                self.tiff_folder_path,
                self.spec_basename,
                scan_number,
            )
        except Exception as error:
            messagebox.showerror("TIFF Search Error", str(error))
            return

        if len(tiff_files) != len(incident_energy_eV):
            messagebox.showerror(
                "Point Mismatch",
                f"SPEC scan has {len(incident_energy_eV)} points, but TIFF folder has {len(tiff_files)} files.",
            )
            return

        mean_i2 = float(np.mean(i2_values))
        self.log(f"Mean i2 across scan: {mean_i2:.4f} (used for pseudo-CPS scaling)")

        spectra = []
        emission_energy_eV = None
        expected_shape = None
        disp_span = None

        for point_index, tiff_path in enumerate(tiff_files):
            try:
                image = load_tiff(tiff_path)
            except Exception as error:
                messagebox.showerror("TIFF Read Error", str(error))
                return

            if expected_shape is None:
                expected_shape = image.shape
            elif image.shape != expected_shape:
                messagebox.showerror("Shape Mismatch", "TIFF shape changed between files.")
                return

            try:
                this_emission_axis, normalized_spectrum, disp_span = process_hamos_image(
                    image=image,
                    i2_value=i2_values[point_index],
                    mean_i2=mean_i2,
                    dispersive_axis=dispersive_axis,
                    roi_low=roi_low,
                    roi_high=roi_high,
                    bad_roi_low=bad_roi_low,
                    bad_roi_high=bad_roi_high,
                    Eb_keV=Eb_keV,
                    Eref_keV=Eref_keV,
                    Pr_pixel=Pr_pixel,
                    Ps_mm=Ps_mm,
                    RC_mm=RC_mm,
                    Ds_mm=Ds_mm,
                    count_time=count_time,
                    dy_per_col=active_dy,
                )
            except Exception as error:
                messagebox.showerror("Processing Error", str(error))
                return

            if emission_energy_eV is None:
                emission_energy_eV = this_emission_axis
            elif not np.allclose(emission_energy_eV, this_emission_axis):
                messagebox.showerror("Axis Error", "Emission energy axis changed.")
                return

            spectra.append(normalized_spectrum)

        hamos_map = np.vstack(spectra)

        incident_order = np.argsort(incident_energy_eV)
        incident_energy_eV = incident_energy_eV[incident_order]
        i2_values = i2_values[incident_order]
        hamos_map = hamos_map[incident_order, :]

        emission_order = np.argsort(emission_energy_eV)
        emission_energy_eV = emission_energy_eV[emission_order]
        hamos_map = hamos_map[:, emission_order]

        self.hamos_map = hamos_map
        self.incident_axis_eV = incident_energy_eV
        self.emission_axis_eV = emission_energy_eV
        self.i2_values = i2_values

        self.metadata = {
            "spec_file": self.spec_file_path,
            "scan_number": int(scan_number),
            "Eb_keV": float(Eb_keV),
            "Pr_pixel": float(Pr_pixel),
            "Eref_keV": float(Eref_keV),
            "roi_low_x": int(roi_low),
            "roi_high_x": int(roi_high),
            "excluded_bad_pixel_low_x": bad_roi_low,
            "excluded_bad_pixel_high_x": bad_roi_high,
            "mean_i2": mean_i2,
            "normalization": "pseudo-CPS: (counts / i2 / second) * mean_i2",
            "curvature_corrected": bool(apply_curve),
            "reference_tiff": self.ref_tiff_path if apply_curve else None,
            "map_shape": list(hamos_map.shape),
        }

        self.log(f"Final RXES map shape: {hamos_map.shape}")
        self.log("=" * 68)

        self._replot()

    def _replot(self):
        if self.hamos_map is None:
            messagebox.showwarning("Warning", "Generate map first.")
            return

        cmap = self.cmap_var.get()
        scale = self.scale_var.get()
        percentile_low = self.pct_low_var.get()
        percentile_high = self.pct_high_var.get()
        plot_choice = self.plot_type_var.get()

        plot_map = self.hamos_map.T

        if plot_choice == "Energy Transfer":
            et_min = self.incident_axis_eV.min() - self.emission_axis_eV.max()
            et_max = self.incident_axis_eV.max() - self.emission_axis_eV.min()
            n_et = len(self.emission_axis_eV)
            et_axis = np.linspace(et_min, et_max, n_et)

            et_map = np.zeros_like(plot_map)
            for j, Ei in enumerate(self.incident_axis_eV):
                col_data = plot_map[:, j]
                et_vals = Ei - self.emission_axis_eV
                et_map[:, j] = np.interp(et_axis, et_vals[::-1], col_data[::-1], left=0.0, right=0.0)

            y_axis_centers = et_axis
            y_label = "Energy Transfer (eV)"
            plot_data = et_map
            title_prefix = " (Energy Transfer)"
        else:
            y_axis_centers = self.emission_axis_eV
            y_label = "Emission Energy (eV)"
            plot_data = plot_map
            title_prefix = " (Emission Energy)"

        positive_values = plot_data[plot_data > 0]

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        incident_edges = axis_edges(self.incident_axis_eV)
        y_edges = axis_edges(y_axis_centers)

        if scale == "log" and positive_values.size > 0:
            vmin = max(np.percentile(positive_values, percentile_low), 1e-15)
            vmax = np.percentile(positive_values, percentile_high)
            image = self.ax.pcolormesh(
                incident_edges,
                y_edges,
                plot_data,
                shading="flat",
                cmap=cmap,
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )
            color_label = "Pseudo-CPS (counts/s scaled to mean i2) (log)"
        else:
            vmin = np.percentile(positive_values, percentile_low) if positive_values.size > 0 else 0.0
            vmax = np.percentile(positive_values, percentile_high) if positive_values.size > 0 else 1.0
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
            image = self.ax.pcolormesh(
                incident_edges,
                y_edges,
                plot_data,
                shading="flat",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            color_label = "Pseudo-CPS (counts/s scaled to mean i2)"

        colorbar = self.fig.colorbar(image, ax=self.ax)
        colorbar.set_label(color_label)

        if plot_choice == "Emission Energy":
            self.ax.set_ylim(y_axis_centers.max(), y_axis_centers.min())
        else:
            self.ax.set_ylim(y_axis_centers.min(), y_axis_centers.max())

        curve_suffix = " [Curvature Corrected]" if (self.metadata and self.metadata.get("curvature_corrected")) else ""
        self.ax.set_title(f"von Hamos RXES Map{title_prefix}{curve_suffix}")
        self.ax.set_xlabel("Incident Energy (eV)")
        self.ax.set_ylabel(y_label)

        self.fig.tight_layout()
        self.canvas.draw()
        self.log(f"Plot updated: {plot_choice}, {scale} scale, colormap={cmap}")

    def _save_data(self):
        if self.hamos_map is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("Compressed NumPy data", "*.npz"), ("CSV table", "*.csv")],
        )
        if not path:
            return
        try:
            if path.endswith(".npz"):
                np.savez_compressed(
                    path,
                    intensity=self.hamos_map,
                    incident_energy_eV=self.incident_axis_eV,
                    emission_energy_eV=self.emission_axis_eV,
                    metadata_json=json.dumps(self.metadata, indent=2),
                )
                self.log(f"Saved NPZ: {path}")
            elif path.endswith(".csv"):
                plot_map = self.hamos_map.T
                with open(path, "w") as file:
                    header = ["EmissionEnergy_eV"] + [f"Ei_{e:.7f}_eV" for e in self.incident_axis_eV]
                    file.write(",".join(header) + "\n")
                    for i, em_en in enumerate(self.emission_axis_eV):
                        row = [f"{em_en:.8f}"] + [f"{plot_map[i, j]:.10e}" for j in range(len(self.incident_axis_eV))]
                        file.write(",".join(row) + "\n")
                self.log(f"Saved CSV: {path}")
        except Exception as error:
            messagebox.showerror("Save Error", str(error))

    def _save_figure(self):
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            self.log(f"Saved figure: {path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = Hamos2DMapGUI(root)
    root.mainloop()