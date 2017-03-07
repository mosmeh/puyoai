#!/bin/bash
cd "$(dirname "$0")"
exec ./duelrecorder "$@" 2> duelrecorder.err

