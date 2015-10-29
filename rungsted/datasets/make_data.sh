#!/bin/sh
IN_DIR=~/Datasets/wsj

for file in $IN_DIR/*.rungsted; do
    name=$(basename $file)
    name=${name%.rungsted}
    echo "Processing $name"
    python rungsted/datasets/conll_to_vw.py $file data/$name.fine.vw --name $name --feature-set honnibal13-groups
    python rungsted/datasets/conll_to_vw.py $file data/$name.coarse.vw --name $name --feature-set honnibal13-groups --coarse

done;
