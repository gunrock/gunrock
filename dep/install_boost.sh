wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.bz2
tar -xjvf boost_1_58_0.tar.bz2
cd boost_1_58_0
sudo ./bootstrap.sh
sudo ./b2 install --prefix=/usr/local
cd ../
sudo rm -rf boost_1_58_0
sudo rm boost_1_58_0.tar.bz2
