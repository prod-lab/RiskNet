#!/bin/bash




# gcloud authentication
export GOOGLE_APPLICATION_CREDENTIALS=/secrets/key-file
gcloud auth activate-service-account --key-file=/secrets/key-file


# echo commands to the terminal output
set -ex

# pass through commands

CMD=("$@")
exec "${CMD[@]}"
