#!/bin/bash
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e
set -x

eval "$(conda shell.bash hook)"
mamba create --name ancient
conda init bash
conda activate ancient
mamba install -q -y make cmake cxx-compiler=1.0 gcc=8.5 gxx=8.5 binutils


umask 0077
# prefix="$HOME/opt/openssh"
prefix="/"
top="$(pwd)"
root="$top/root"
build="$top/build"
export CPPFLAGS="-I$root/include -L."
rm -rf "$root" "$build"
mkdir -p "$root" "$build"
mkdir dist
cd dist
wget https://www.zlib.net/fossils/zlib-1.2.11.tar.gz
wget https://github.com/openssl/openssl/releases/download/OpenSSL_1_0_2n/openssl-1.0.2n.tar.gz
wget https://cdn.openbsd.org/pub/OpenBSD/OpenSSH/portable/openssh-7.4p1.tar.gz
cd ..



gzip -dc dist/zlib-*.tar.gz |(cd "$build" && tar xf -)
cd "$build"/zlib-*
./configure --prefix="$root" --static
make
make install
cd "$top"


gzip -dc dist/openssl-*.tar.gz |(cd "$build" && tar xf -)
cd "$build"/openssl-*
./config --prefix="$root" no-shared
make
make install
cd "$top"


gzip -dc dist/openssh-*.tar.gz |(cd "$build" && tar xf -)
cd "$build"/openssh-*
cp -p "$root"/lib/*.a .
# TODO config
./configure --prefix="$prefix" --with-privsep-user=nobody --with-privsep-path="$prefix/var/empty"
sed -i '/LOCKED_PASSWD_PREFIX/d' config.h
make -j 64
make install-nokeys

cd "$top"
rm -rf "$root" "$build" dist
