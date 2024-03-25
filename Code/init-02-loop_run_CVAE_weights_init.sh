bash
conda activate tf231

#!/bin/bash
for i in {90..99}
do
   echo "Running it: $i"
   python init-01-run_CVAE_weights_init.py $i
done