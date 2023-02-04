#!/usr/bin/bash
# bring all images in same physical space
# 1. zeroout the origin
# 2. reorient all images to same orientation say ASL
dir='data/cleaned/'
mkdir -p data/origin_zeroed
c3d ${dir}*.nii.gz -foreach -origin 0x0x0mm -endfor -oo data/origin_zeroed/data_sample%03d.nii.gz

c3d data/origin_zeroed/data_sample*.nii.gz -foreach -orient ASL -endfor -oo data/origin_zeroed/data_sample%03d.nii.gz
# generate mean shape and binarize
c3d data/origin_zeroed/data_sample*.nii.gz -mean -binarize -o data/mean_template.nii.gz