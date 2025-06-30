#!/bin/bash
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

printenv SSH_PUBLIC_KEY >> $HOME/.ssh/authorized_keys
$HOME/opt/openssh/sbin/sshd -D -e -f $HOME/.ssh/sshd_config
