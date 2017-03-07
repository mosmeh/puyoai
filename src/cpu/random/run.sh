#!/bin/bash
cd "$(dirname "$0")"
exec ./random "$@" 2> random.err

