#!/bin/bash

if [ -z "$device" ];
then 
    device=0 
fi
if [ -z "$knn" ];
then 
    knn=gunrock
fi

json=$knn

mkdir -p JSON/$json

# Ordered by size of test

# KDDCUPB74d (KDD 145K, 74d, k=30, eps=5, min-pts=5)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file /data/clustering_dataset/KDDCUP04Bio/KDDCUP04Bio.txt --k=30 --eps=5 --min-pts=5 --NUM-THREADS=1024 --transpose=true --quick=true --jsonfile JSON/$json/kdd145k74d.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file /data/clustering_dataset/KDDCUP04Bio/KDDCUP04Bio.txt --k=30 --eps=5 --min-pts=5 --NUM-THREADS=1024 --transpose=true --quick=true --jsonfile JSON/$json/kdd145k74d.json 

# 3DRN (434K, 3d, k=30, eps=12, min-pts=15)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file /data/clustering_dataset/3DRN/3D_spatial_network.txt --k=30 --eps=12 --min-pts=15 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/3DRN.json "
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file /data/clustering_dataset/3DRN/3D_spatial_network.txt --k=30 --eps=12 --min-pts=15 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/3DRN.json 

# DGF5L11d (Font 572K, 11d, k=30, eps=12, min-pts=15)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Font/dim11/DGF572K11d --k=30 --eps=12 --min-pts=15 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/DGF572K11d.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Font/dim11/DGF572K11d --k=30 --eps=12 --min-pts=15 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/DGF572K11d.json

# MPAH1.5M9d (Halo 1M, 9d, k=40, eps=15, min-pts=20)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Halo/dim9/MPAH1M9d --k=40 --eps=15 --min-pts=20 --quick --jsonfile JSON/$json/MPAH1M9d"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Halo/dim9/MPAH1M9d --k=40 --eps=15 --min-pts=20 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/MPAH1M9d

# MPAH1.5M9d (Halo 1.5M, 9d, k=40, eps=15, min-pts=20)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Halo/dim9/MPAH1.5M9d --k=40 --eps=15 --min-pts=20 --quick --jsonfile JSON/$json/MPAH1.5M9d"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Halo/dim9/MPAH1.5M9d --k=40 --eps=15 --min-pts=20 --quick --NUM-THREADS=1024 --transpose=true --jsonfile JSON/$json/MPAH1.5M9d

# DGF2M11d (Font 2.1M, 11d, k=50, eps=20, min-pts=25)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Font/dim11/DGF2M11d --k=50 --eps=20 --min-pts=25 --quick --jsonfile JSON/$json/DGF2M11d.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Font/dim11/DGF2M11d --k=50 --eps=20 --min-pts=25 --quick --NUM-THREADS=1024 --transpose=true --jsonfile JSON/$json/DGF2M11d.json

# MPAGB23K3d (Berton 8M, 3d, k=30, eps=12, min-pts=15)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Berton/dim3/MPAGB23K3d_0 --k=30 --eps=12 --min-pts=15 --quick --jsonfile JSON/$json/MPAGB23K3d_0.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Berton/dim3/MPAGB23K3d_0 --k=30 --eps=12 --min-pts=15 --quick --NUM-THREADS=1024 --transpose=true --jsonfile JSON/$json/MPAGB23K3d_0.json

# MPAGB8M3d (Berton 8M, 3d, k=50, eps=20, min-pts=25)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Berton/dim3/MPAGB8M3d_0 --k=50 --eps=20 --min-pts=25 --quick --jsonfile JSON/$json/MPAGB8M3d_0.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Berton/dim3/MPAGB8M3d_0 --k=50 --eps=20 --min-pts=25 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/MPAGB8M3d_0.json

# DGB8M3d (Bower 8.7M, 3d, k=50, eps=20, min-pts=25)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Bower/dim3/DGB8M3d_0 --k=50 --eps=20 --min-pts=25 --quick --jsonfile JSON/$json/DGB8M3d_0.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/Bower/dim3/DGB8M3d_0 --k=50 --eps=20 --min-pts=25 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/DGB8M3d_0.json 

# FOF11M3d (FOF 11.4M, 3d, k=30, eps=12, min-pts=15)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF11M3d --k=30 --eps=12 --min-pts=15 --quick --jsonfile JSON/$json/FOF11M3d.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF11M3d --k=30 --eps=12 --min-pts=15 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/FOF11M3d.json 

# FOF20M3d (FOF 20M, 3d, k=30, eps=12, min-pts=15)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF20M3d --k=30 --eps=12 --min-pts=15 --quick --jsonfile JSON/$json/FOF20M3d.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF20M3d --k=30 --eps=12 --min-pts=15 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/FOF20M3d.json 

# FOF22M3d (FOF 22M, 3d, k=30, eps=12, min-pts=15)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF22M3d --k=30 --eps=12 --min-pts=15 --quick --jsonfile JSON/$json/FOF22M3d.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF22M3d --k=30 --eps=12 --min-pts=15 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/FOF22M3d.json 

# FOF25M3d (FOF 25M, 3d, k=30, eps=12, min-pts=15)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF25M3d --k=30 --eps=12 --min-pts=15 --quick --jsonfile JSON/$json/FOF25M3d.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF25M3d --k=30 --eps=12 --min-pts=15 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/FOF25M3d.json 

# FOF30M3d (FOF 30M, 3d, k=30, eps=12, min-pts=15)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF30M3d --k=30 --eps=12 --min-pts=15 --quick --jsonfile JSON/$json/FOF30M3d.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF30M3d --k=30 --eps=12 --min-pts=15 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/FOF30M3d.json 

# FOF57M3d (FOF 57M, 3d, k=30, eps=12, min-pts=15)
echo "./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF57M3d --k=30 --eps=12 --min-pts=15 --quick --jsonfile JSON/$json/FOF57M3d.json"
./bin/test_snn_10.2_x86_64 --device=$device --knn-version=$knn market --labels-file ~/clustering_dataset/FOF/FOF57M3d --k=30 --eps=12 --min-pts=15 --NUM-THREADS=1024 --transpose=true --quick --jsonfile JSON/$json/FOF57M3d.json 


