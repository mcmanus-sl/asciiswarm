#!/bin/bash
cd "$(dirname "$0")/.."
git pull upstream main
git push origin main
