if [ -z "$1" ]
  then
    echo "error: usage: ./install_metis.sh 32 or 64"

  else
    wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
    gunzip metis-5.1.0.tar.gz
    tar -xvf metis-5.1.0.tar
    cd metis-5.1.0
    if [ $1 == 64 ]; then
      #64-bit metis installation
      cd include
      sed -i.bak 's/#define IDXTYPEWIDTH 32/#define IDXTYPEWIDTH 64/' ./metis.h
      cd ../
    fi
    sudo make config
    sudo make
    sudo make install
    cd ../
    rm metis-5.1.0.tar
    sudo rm -rf metis-5.1.0
fi
