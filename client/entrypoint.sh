#!/bin/bash

# Run your script and capture the exit status
cd /app || exit 1
python flwr_dr_client.py "$@"
#python flwr_dr_client_eval.py "$@"
#python flwr_dr_client_only_hpc.py "$@"
EXIT_STATUS=$?

# Print a message indicating the script has finished
echo "Script has completed with exit status $EXIT_STATUS."

# Check the exit status and keep the container open with a bash shell
if [ $EXIT_STATUS -eq 0 ]; then
  echo "Exiting the script normally. Opening bash shell..."
  exec /bin/bash
else
  echo "Script failed with exit status $EXIT_STATUS. Exiting."
  exit $EXIT_STATUS
fi

