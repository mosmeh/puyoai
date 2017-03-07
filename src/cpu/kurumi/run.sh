#!/bin/bash
cd "$(dirname "$0")"
exec ./kurumi "$@" 2> kurumi.err

