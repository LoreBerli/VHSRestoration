#!/bin/bash

unpackedVideo='../dataset/unpacked_video'
frameDir="../dataset/frames"
frameLR="../dataset/training_set/LR_"


if [ ! -d "$unpackedVideo" ];
  then
     echo The directory unpackeVideo not exists, the script ends its execution.
     exit
fi

#if [ -d "$frameDir" ];
#  then
#     echo The directory frames is already exists.
#  else
#     echo The directory frames it was created
 #    mkdir $frameDir
#fi


for res in HDReady FullHD
   do
      if [ -d "$frameLR"$res ];
        then
           echo The directory LR_$res is already exists.
        else
           echo The directory LR_$res it was created.
           mkdir $frameLR$res
      fi
   done

for res in HDReady FullHD
  do
    for set in set1 set2 set3 set4 set5
      do
        if [ -d "$frameLR"$res"/"$set ]
          then
            echo The directory LR_$res/$set is already exists.
          else
            echo The directory LR_$res/$set it was created.
            mkdir $frameLR$res/$set
        fi
      done
  done

<<comment
for res in HDReady FullHD
   do
      for source in $(ls $unpackedVideo|grep .$res)
         do
           echo source:$unpackedVideo'/'$source
           echo dest:$frameLR$res'/'$source
           mkdir $frameLR$res'/'$source
           ffmpeg -i $unpackedVideo'/'$source -f image2 -vf fps=fps=1/10 $frameLR$res'/'$source'/'"_%3d.jpg"
         done
   done
comment
#unpackedVideo='../dataset/unpacked_video'
pwd
for res in HDReady FullHD
   do
     for set in set1 set2 set3 set4 set5
        do
          for source in $(ls $unpackedVideo"/"$res"/"$set)
             do
             echo source:$unpackedVideo"/"$res"/"$set"/"$source
             echo dest:$frameLR$res"/"$set"/"$source
             mkdir $frameLR$res'/'$set'/'$source
             time=$(ffprobe -i $unpackedVideo"/"$res"/"$set"/"$source -show_entries format=duration -v quiet -of csv="p=0")
             time=${time/.*}
             echo "valore di time: $time"
             if [ $time -ge 1050 ]
               then
                 echo ciao
                 ffmpeg -i $unpackedVideo"/"$res"/"$set"/"$source -r 0.1 $frameLR$res"/"$set"/"$source"/""_%3d.jpg"
            fi
            if [ $time -ge 754 ] && [ $time -lt 1050 ]
               then
                 echo hello
                 ffmpeg -i $unpackedVideo"/"$res"/"$set"/"$source -r 0.125 $frameLR$res"/"$set"/"$source"/""_%3d.jpg"
            fi
            if [ $time -ge 605 ] && [ $time -le 754 ]
               then
                 echo pippo
                 ffmpeg -i $unpackedVideo"/"$res"/"$set"/"$source -r 0.166 $frameLR$res"/"$set"/"$source"/""_%3d.jpg"
            fi
            if [ $time -ge 520 ] && [ $time -le 605 ]
               then
                 echo pluto
                 ffmpeg -i $unpackedVideo"/"$res"/"$set"/"$source -r 0.2 $frameLR$res"/"$set"/"$source"/""_%3d.jpg"
            fi
            if [ $time -lt 520 ]
             then
               echo mercurio
               ffmpeg -i $unpackedVideo"/"$res"/"$set"/"$source -r 0.5 $frameLR$res"/"$set"/"$source"/""_%3d.jpg"
             fi
             #ffmpeg -i $unpackedVideo"/"$res"/"$set"/"$source -f image2 -vf fps=fps=1/10 $frameLR$res"/"$set"/"$source"/""_%3d.jpg"
             done
        done
     done













