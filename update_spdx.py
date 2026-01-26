#!/usr/bin/env python3
"""
Add SPDX identifier and update copyright year in Rust source files.
"""

import os
import glob
import re

# The new header format with SPDX
NEW_HEADER = """// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Existing header pattern (without SPDX)
OLD_HEADER_PATTERN = re.compile(
    r'^// SochDB - LLM-Optimized Embedded Database\n'
    r'// Copyright \(C\) 202[56] Sushanth.*?\n'
    r'//\n'
    r'// This program is free software:.*?'
    r'// along with this program\. If not, see <https://www\.gnu\.org/licenses/>.\n',
    re.MULTILINE | re.DOTALL
)

def update_file(filepath):
    """Update a single file with SPDX identifier."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already has SPDX identifier
        if content.startswith('// SPDX-License-Identifier:'):
            # Already updated, but maybe need to update copyright year
            if 'Copyright (C) 2025' in content:
                content = content.replace('Copyright (C) 2025', 'Copyright (C) 2026', 1)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return 'updated_year'
            return 'already_has_spdx'
        
        # Replace old header with new one
        new_content = OLD_HEADER_PATTERN.sub(NEW_HEADER, content, count=1)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return 'updated'
        
        return 'no_header'
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 'error'

def main():
    # Find all .rs files in sochdb directory
    rust_files = []
    for root, dirs, files in os.walk('/Users/sushanth/sochdb/sochdb'):
        # Skip target directory
        if 'target' in root:
            continue
        for file in files:
            if file.endswith('.rs'):
                rust_files.append(os.path.join(root, file))
    
    stats = {
        'updated': 0,
        'updated_year': 0,
        'already_has_spdx': 0,
        'no_header': 0,
        'error': 0
    }
    
    for filepath in rust_files:
        result = update_file(filepath)
        stats[result] += 1
        if result in ['updated', 'updated_year']:
            print(f"Updated: {filepath}")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Files updated with SPDX: {stats['updated']}")
    print(f"  Files with year updated: {stats['updated_year']}")
    print(f"  Files already had SPDX: {stats['already_has_spdx']}")
    print(f"  Files without header: {stats['no_header']}")
    print(f"  Errors: {stats['error']}")
    print(f"  Total files processed: {len(rust_files)}")
    print("="*60)

if __name__ == '__main__':
    main()
