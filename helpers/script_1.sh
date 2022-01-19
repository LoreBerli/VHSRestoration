#!/bin/sh

datasetDir='../dataset'
unpackedVideo="../dataset/unpacked_video"
packedVideo="../dataset/packed_video"


if [ -d $datasetDir ];
   then
      echo The directory dataset is already exists.
   else
      echo The directory dataset it was created.
      mkdir $datasetDir
fi

if [ -d $packedVideo ];
   then
      echo The directory packedVideo  is already exists.
   else
      echo The directory packedVideo it was created.
      mkdir $packedVideo
fi

if [ -d $unpackedVideo ];
  then
     echo The directory unpackedVideo is already exists.
  else
     echo The directory unpackedVideo it was created.
     mkdir $unpackedVideo
fi

cd $packedVideo

rarFile=$(ls |grep .rar)
for file in $rarFile
   do
     mv "$file" `echo $file | tr ' ' '_'` # rimpiazza gli spazzi vuoti con _ nei nomi dei file in packedVideo
     echo $file
   done

cd ..

echo $rarFile
for source in $rarFile
   do
     echo source:$packedVideo'/'$source
     echo dest:$unpackedVideo'/'$source
     unrar e $packedVideo'/'$source $unpackedVideo'/'
   done

cd $unpackedVideo

for file in *
   do
     mv "$file" `echo $file | tr ' ' '_'` ;
   done
com


for dir in HDReady FullHD
   do
     mkdir $dir
     for set in set1 set2 set3 set4 set5
       do
         mkdir $dir/$set
       done
    done

for dir in HDReady FullHD
   do
     for set in set1 set2 set3 set4 set5
       do
         getList=$(ls |grep $dir |grep $set)#filtra i video per risoluzione e set
         for file in $getList
            do
              mv "$file" $dir/$set #sposta il video nella cartella
            done

       done
   done
