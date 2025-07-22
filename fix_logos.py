#!/usr/bin/env python3
"""
Fix logo file names to match what the code expects.
This script renames the logo files to the correct names.
"""

import os
import shutil
from pathlib import Path

def fix_logo_names():
    """Rename logo files to match expected names in the code."""
    assets_dir = Path("sed_demo/assets")
    
    if not assets_dir.exists():
        print("❌ sed_demo/assets directory not found!")
        return False
    
    # Mapping of current names to expected names
    logo_mappings = {
        "Rai4s_banner.png": "ai4s_banner.png",
        "Rsurrey_logo.png": "surrey_logo.png", 
        "REPSRC_logo.png": "EPSRC_logo.png",
        "RCVSSP_logo.png": "CVSSP_logo.png"
    }
    
    print("🔧 Fixing logo file names...")
    
    fixed_count = 0
    for old_name, new_name in logo_mappings.items():
        old_path = assets_dir / old_name
        new_path = assets_dir / new_name
        
        if old_path.exists():
            if new_path.exists():
                print(f"✅ {new_name} already exists")
            else:
                try:
                    shutil.copy2(old_path, new_path)
                    print(f"✅ Renamed {old_name} → {new_name}")
                    fixed_count += 1
                except Exception as e:
                    print(f"❌ Failed to rename {old_name}: {e}")
        else:
            print(f"⚠️  {old_name} not found")
    
    print(f"\n📊 Fixed {fixed_count} logo files")
    return fixed_count > 0

if __name__ == "__main__":
    fix_logo_names()