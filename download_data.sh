#!/bin/bash -e
# WIKI
PROJECT_DIR=`pwd`
mkdir -p dataset && cd dataset
mkdir -p wiki && cd wiki
mkdir -p index && mkdir -p value
if [ ! -f 'value/enwiki-20230101-pages-articles-multistream1.xml-p1p41242' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream1.xml-p1p41242.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream1.xml-p1p41242.bz2
  mv enwiki-20230101-pages-articles-multistream1.xml-p1p41242 value/
fi
if [ ! -f 'index/enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242.bz2
  mv enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242 index/
fi

if [ ! -f 'value/enwiki-20230101-pages-articles-multistream2.xml-p41243p151573' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream2.xml-p41243p151573.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream2.xml-p41243p151573.bz2
  mv enwiki-20230101-pages-articles-multistream2.xml-p41243p151573 value/
fi
if [ ! -f 'index/enwiki-20230101-pages-articles-multistream-index2.txt-p41243p151573' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream-index2.txt-p41243p151573.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream-index2.txt-p41243p151573.bz2
  mv enwiki-20230101-pages-articles-multistream-index2.txt-p41243p151573 index/
fi

if [ ! -f 'value/enwiki-20230101-pages-articles-multistream3.xml-p151574p311329' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream3.xml-p151574p311329.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream3.xml-p151574p311329.bz2
  mv enwiki-20230101-pages-articles-multistream3.xml-p151574p311329 value/
fi
if [ ! -f 'index/enwiki-20230101-pages-articles-multistream-index3.txt-p151574p311329' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream-index3.txt-p151574p311329.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream-index3.txt-p151574p311329.bz2
  mv enwiki-20230101-pages-articles-multistream-index3.txt-p151574p311329 index/
fi

if [ ! -f 'value/enwiki-20230101-pages-articles-multistream4.xml-p311330p558391' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream4.xml-p311330p558391.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream4.xml-p311330p558391.bz2
  mv enwiki-20230101-pages-articles-multistream4.xml-p311330p558391 value/
fi
if [ ! -f 'index/enwiki-20230101-pages-articles-multistream-index4.txt-p311330p558391' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream-index4.txt-p311330p558391.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream-index4.txt-p311330p558391.bz2
  mv enwiki-20230101-pages-articles-multistream-index4.txt-p311330p558391 index/
fi

if [ ! -f 'value/enwiki-20230101-pages-articles-multistream5.xml-p558392p958045.bz2' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream5.xml-p558392p958045.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream5.xml-p558392p958045.bz2
  mv enwiki-20230101-pages-articles-multistream5.xml-p558392p958045.bz2 value/
fi
if [ ! -f 'index/enwiki-20230101-pages-articles-multistream-index5.txt-p558392p958045.bz2' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream-index5.txt-p558392p958045.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream-index5.txt-p558392p958045.bz2
  mv enwiki-20230101-pages-articles-multistream-index5.txt-p558392p958045.bz2 index/
fi

if [ ! -f 'value/enwiki-20230101-pages-articles-multistream6.xml-p958046p1483661.bz2' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream6.xml-p958046p1483661.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream6.xml-p958046p1483661.bz2
  mv enwiki-20230101-pages-articles-multistream6.xml-p958046p1483661.bz2 value/
fi
if [ ! -f 'index/enwiki-20230101-pages-articles-multistream-index6.txt-p958046p1483661.bz2' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream-index6.txt-p958046p1483661.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream-index6.txt-p958046p1483661.bz2
  mv enwiki-20230101-pages-articles-multistream-index6.txt-p958046p1483661.bz2 index/
fi

cd $PROJECT_DIR

# YCSB
cd YCSB-C 
make clean
make
cd $PROJECT_DIR
mkdir -p dataset && cd dataset
mkdir -p ycsb && cd ycsb
rm workloada.txt
$PROJECT_DIR/YCSB-C/ycsbc -db basic -threads 4 -P $PROJECT_DIR/YCSB-C/workloads/workloada.spec > workloada.txt

