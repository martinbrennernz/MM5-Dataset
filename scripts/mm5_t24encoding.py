#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#  mm5_t24encoding.py
#
#  LICENSE:
#    This file is distributed under the Creative Commons
#    Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
#    https://creativecommons.org/licenses/by-nc/4.0/
#
#    You are free to:
#      • Share — copy and redistribute the material in any medium or format
#      • Adapt — remix, transform, and build upon the material
#    under the following terms:
#      • Attribution — You must give appropriate credit, provide a link to
#        the license, and indicate if changes were made.
#      • NonCommercial — You may not use the material for commercial purposes.
#
#
#  DISCLAIMER:
#    This code is provided “AS IS,” without warranties or conditions of any kind,
#    express or implied. Use it at your own risk.
# -----------------------------------------------------------------------------

import numpy as np

class ColourMap:
    def __init__(self):
        """
        Initialize the ColourMap by generating the color gradient and clipping it.
        """
        self.full_gradient = []
        self.clipped_gradient = []
        self.generate_gradient()

    def generate_gradient(self):
        """
        Generates the full color gradient (19 segments, each with 256 steps),
        then clips the first 20 and last 20 entries to remove excessively dark tones.
        """
        self.full_gradient.clear()

        # ---- Dark / Cool Region ----
        self.add_color_range(0,   0,   0,    75,  0,   130)     # (1) Black       → Purple
        self.add_color_range(75,  0,   130,  25,  25,  112)     # (2) Purple      → Navy
        self.add_color_range(25,  25,  112,  0,   0,   139)     # (3) Navy        → Deep Blue
        self.add_color_range(0,   0,   139,  0,   0,   255)     # (4) Deep Blue   → Blue
        self.add_color_range(0,   0,   255,  200, 200, 200)     # (5) Blue        → Grey
        self.add_color_range(200, 200, 200,  0,   128, 128)     # (6) Grey        → Teal
        self.add_color_range(0,   128, 128,  127, 255, 212)     # (7) Teal        → Aquamarine
        self.add_color_range(127, 255, 212,  0,   255, 255)     # (8) Aquamarine  → Cyan

        # ---- Mid / Warmer Region ----
        self.add_color_range(0,   255, 255,  0,   255, 0)       # (9)  Cyan       → Green
        self.add_color_range(0,   255, 0,    192, 255, 0)       # (10) Green      → Lime Green
        self.add_color_range(192, 255, 0,    255, 215, 0)       # (11) Lime Green → Gold
        self.add_color_range(255, 215, 0,    255, 255, 0)       # (12) Gold       → Yellow

        # ---- Hot Region ----
        self.add_color_range(255, 255, 0,    255, 165, 0)       # (13) Yellow     → Orange
        self.add_color_range(255, 165, 0,    255, 100, 0)       # (14) Orange     → Dark Orange
        self.add_color_range(255, 100, 0,    200, 0,   0)       # (15) Dark Orange→ Dark Red
        self.add_color_range(200, 0,   0,    255, 0,   0)       # (16) Dark Red   → Red
        self.add_color_range(255, 0,   0,    255, 0,   255)     # (17) Red        → Magenta
        self.add_color_range(255, 0,   255,  255, 20,  147)     # (18) Magenta    → Rose
        self.add_color_range(255, 20,  147,  255, 105, 180)     # (19) Rose       → Light Pink
        self.add_color_range(255, 105, 180, 255, 255, 255)      # (20) Light Pink → White


        # Clipping not needed for given gradients
        clip_amount = 0
        if clip_amount==0:
            self.clipped_gradient = self.full_gradient
        else: 
            self.clipped_gradient = self.full_gradient[clip_amount:-clip_amount]

    def add_color_range(self, start_r, start_g, start_b, end_r, end_g, end_b, steps=None):
        """
        Adds a linear interpolation from (start_r, start_g, start_b) to (end_r, end_g, end_b)
        into the full_gradient list.
        
        If 'steps' is provided (e.g. steps=256), the range is stretched to that many steps.
        If 'steps' is not provided (None), the function calculates the number of steps
        based on the maximum channel difference so that only the distinct colors are added.
        """
        # If steps is not provided, calculate the exact number of distinct steps.
        # For each channel, the distinct steps available is the difference plus one.
        if steps is None:
            steps = max(abs(end_r - start_r), abs(end_g - start_g), abs(end_b - start_b)) + 1

        # Compute the interpolated values.
        for i in range(steps):
            r = start_r + int((end_r - start_r) * i / (steps - 1))
            g = start_g + int((end_g - start_g) * i / (steps - 1))
            b = start_b + int((end_b - start_b) * i / (steps - 1))
            self.full_gradient.append([r, g, b])


    def get_temperature_color(self, temp_celsius):
        """
        Maps a temperature value (in Celsius) to the corresponding RGB color
        by computing an index in the clipped gradient. This replicates
        the logic in the C++ reference exactly.
        """

        # --- Define the breakpoints and resolutions (matching the C++ code) ---
        f0max = 6
        f0res = 0.01  # step for range 3°C to 14°C
        # Primary temperature range:
        #   f1idx: Base index in the gradient corresponding to the start of the primary range.
        #   f1min: Lower bound of the primary temperature range.
        #   f1max: Upper bound of the primary temperature range.
        #   f1res: Resolution (°C per step) for the primary range.
        f1idx = 700
        f1min = 12
        f1max = 20
        f1res = 0.008

        # Moderate high temperature range:
        #   f2max: Upper bound of the moderate high temperature range.
        #   f2res: Resolution (°C per step) for this temperature range.
        f2max = 35
        f2res = 0.025

        # High temperature range:
        #   f3max: Upper bound of the high temperature range.
        #   f3res: Resolution (°C per step) for the high temperature range.
        f3max = 100
        f3res = 0.1

        # Compute the index by piecewise logic
        if f1min <= temp_celsius < f1max:
            # 14°C <= T < 30°C
            index = f1idx + int((temp_celsius - f1min) / f1res)
        elif f1max <= temp_celsius <= f2max:
            # 30°C <= T <= 40°C
            range_in_f1 = int((f1max - f1min) / f1res)
            index = f1idx + range_in_f1 + int((temp_celsius - f1max) / f2res)
        elif f2max < temp_celsius <= f3max:
            # 40°C < T <= 100°C
            range_in_f1 = int((f1max - f1min) / f1res)
            range_in_f2 = int((f2max - f1max) / f2res)
            index = f1idx + range_in_f1 + range_in_f2 + int((temp_celsius - f2max) / f3res)
        elif temp_celsius > f3max:
            # T > 100°C
            range_in_f1 = int((f1max - f1min) / f1res)
            range_in_f2 = int((f2max - f1max) / f2res)
            range_in_f3 = int((f3max - f2max) / f3res)
            index = (f1idx
                     + range_in_f1
                     + range_in_f2
                     + range_in_f3
                     + int((temp_celsius - f3max) / fminres))
        elif temp_celsius < f1min and temp_celsius >= f0max:
            # 3°C <= T < 14°C
            index = f1idx - int((f1min - temp_celsius) / f0res)
        elif temp_celsius < f0max:
            # T < 3°C
            # step 1: from 14°C down to 3°C
            step_below_f1min = int((f1min - f0max) / f0res)
            # step 2: from 3°C down to T
            index = (f1idx
                     - step_below_f1min
                     - int((f0max - temp_celsius) / fminres))
        else:
            # Fallback for any unforeseen condition
            return [0, 0, 0]

        # --- Clamp the index to valid bounds ---
        if index < 0 or index >= len(self.clipped_gradient):
            return [0, 0, 0]  # out-of-range => black

        return self.clipped_gradient[index]
        
