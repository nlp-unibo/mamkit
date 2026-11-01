#!/usr/bin/env bash
set -euo pipefail

rm -rf build/html
sphinx-build -W --keep-going -b html source build/html
touch build/html/.nojekyll
