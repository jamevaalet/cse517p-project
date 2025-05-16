#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Jame Vaalet,jvaalet" > submit/team.txt

# train model
#python src/myprogram.py train --work_dir work

# Copy predictions for example data from output/ directory
echo "Copying predictions from output/pred.txt to submit/pred.txt..."
if [ -f "output/pred.txt" ]; then
    cp output/pred.txt submit/pred.txt
else
    echo "ERROR: output/pred.txt not found."
    echo "Please generate predictions first, e.g., by running: bash train_test_model.sh --test"
    exit 1
fi

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# submit requirements.txt
cp requirements.txt submit/requirements.txt

# make zip file
zip -r submit.zip submit
