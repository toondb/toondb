#!/usr/bin/env python3
"""Update license headers from Apache 2.0 to AGPL-3.0"""

import os
import glob

# Handle partially modified headers (from failed sed)
OLD_HEADER_PARTIAL = """// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License."""

OLD_HEADER_ORIGINAL = """// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License."""

NEW_HEADER = """// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2025 Sushanth (https://github.com/sushanthpy)
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
// along with this program. If not, see <https://www.gnu.org/licenses/>."""

def main():
    base_dir = "/Users/sushanth/sochdb/sochdb"
    count = 0
    
    for filepath in glob.glob(f"{base_dir}/**/*.rs", recursive=True):
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            updated = False
            if OLD_HEADER_PARTIAL in content:
                content = content.replace(OLD_HEADER_PARTIAL, NEW_HEADER)
                updated = True
            elif OLD_HEADER_ORIGINAL in content:
                content = content.replace(OLD_HEADER_ORIGINAL, NEW_HEADER)
                updated = True
            
            if updated:
                with open(filepath, 'w') as f:
                    f.write(content)
                count += 1
                print(f"Updated: {filepath}")
        except Exception as e:
            print(f"Error with {filepath}: {e}")

    print(f"\nTotal files updated: {count}")

if __name__ == "__main__":
    main()
