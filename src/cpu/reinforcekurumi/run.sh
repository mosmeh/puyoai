#!/bin/bash
cd "$(dirname "$0")"
exec ./reinforcekurumi "$@" 2> reinforcekurumi.err

