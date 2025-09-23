#!/bin/bash
set -e
pushd src/dojo/tasks/mlebench
git clone --depth 1 https://github.com/openai/mle-bench.git
pushd mle-bench
git fetch --depth 1 origin d0f60ad0d3b2287469ac3c8ac9767330c928c980
git checkout FETCH_HEAD
git lfs fetch --all
git lfs pull
sed -i '29s/.*/cache = dc.Cache(size_limit=2\*\*26)/g' mlebench/data.py
pip install -e .
popd
popd
