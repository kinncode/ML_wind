#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一鍵執行 NREL 5MW 資源評估與發電預測。"""
import runpy

for script in ["01_resource_assessment.py", "02_forecast_pipeline.py"]:
    print("\n" + "=" * 70)
    print(f"執行 {script}")
    print("=" * 70)
    runpy.run_path(script, run_name="__main__")

print("\n全部完成。見 REPORT.md 與 results/")
