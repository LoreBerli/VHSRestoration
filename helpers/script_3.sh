#!/bin/bash

originalVideo='../dataset/original_video'
#frameDir="../dataset/frames"
frameHQ="../dataset/training_set/HQ"



if [ ! -d "$originalVideo" ];
  then
     echo The directory originalVideo not exists, the script ends its execution.
     exit
fi

<<comment
if [ -d "$frameDir" ];
  then
     echo The directory frames is already exists.
  else
     echo The directory frames it was created
     mkdir $frameDir
fi
comment

if [ -d "$frameHQ" ];
  then
     echo The directory HQ is already exists.
  else
     echo The directory HQ it was created
     mkdir $frameHQ
fi

cd $originalVideo
for file in *;
   do
     mv "$file" `echo $file | tr ' ' '_'` ;
   done
cd ..


for source in $(ls $originalVideo|grep .mp4)
   do
     echo source:$originalVideo'/'$source
     echo dest:$frameHQ'/'$source
     mkdir $frameHQ'/'$source
     #ffmpeg -i $originalVideo'/'$source -vf select='between(n\,0\,50)'  -vsync vfr -q:v 1 $frameHQ'/'$source'/'"_%3d.jpg"
     time=$(ffprobe -i $originalVideo'/'$source  -show_entries format=duration -v quiet -of csv="p=0")
     time=${time/.*}
     echo "valore di time: $time"
     if [ $time -ge 1050 ]
       then
         echo ciao
         ffmpeg -i $originalVideo'/'$source -r 0.1 $frameHQ'/'$source'/'"_%3d.jpg"
    fi
    if [ $time -ge 754 ] && [ $time -lt 1050 ]
       then
         echo hello
         ffmpeg -i $originalVideo'/'$source -r 0.125 $frameHQ'/'$source'/'"_%3d.jpg"
    fi
    if [ $time -ge 605 ] && [ $time -le 754 ]
       then
         echo pippo
         ffmpeg -i $originalVideo'/'$source -r 0.166 $frameHQ'/'$source'/'"_%3d.jpg"
    fi
    if [ $time -ge 520 ] && [ $time -le 605 ]
       then
         echo pluto
         ffmpeg -i $originalVideo'/'$source -r 0.2 $frameHQ'/'$source'/'"_%3d.jpg"
    fi
    if [ $time -lt 520 ]
     then
       echo mercurio
       ffmpeg  -i $originalVideo'/'$source -r 0.5 $frameHQ'/'$source'/'"_%3d.jpg"
     fi
   done

