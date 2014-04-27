#!/bin/bash

get_state_count() {
  sed -n 's:.*States\: *\([0-9]*\).*:\1:p' $1
}

get_transition_count() {
  sed -n 's:.*Transitions\: *\([0-9]*\).*:\1:p' $1
}

get_model_checking_time() {
 sed -n 's:.*Time for \(model checking\|transient probability computation\)\: \([0-9,\.]*\) seconds.*:\2:p' $1
}

get_model_construction_time() {
  sed -n 's:.*Time for model construction\: \([0-9,\.]*\) seconds.*:\1:p' $1
}

bench_one() {
  times=$1
  mod_file=$2
  pro_file=$3
  pro=$4
  opt=$5

  tr="-tr"
  for i in $(eval echo "{1..$times}"); do
    if [[ $pro = $tr* ]]; then
      $(./prism.bat $mod_file $pro $opt -s &> prism_bench.out)
    else
      $(./prism.bat $mod_file $pro_file $pro $opt -s &> prism_bench.out)
    fi
   
    if [[ $i -eq 1 ]]; then
      echo $(get_state_count prism_bench.out)
      echo $(get_transition_count prism_bench.out)
    fi 

    echo $(get_model_checking_time prism_bench.out)
  done

  for i in $(eval echo "{1..$times}"); do
    if [[ $pro = $tr* ]]; then
      $(./prism.bat -opencl $mod_file $pro $opt -s &> prism_bench.out)
    else
      $(./prism.bat -opencl $mod_file $pro_file $pro $opt -s &> prism_bench.out)
    fi
    
    echo $(get_model_checking_time prism_bench.out)
  done
}

times=$1
echo $times
while read model; do
  read properties
  read prop_count
  props=()
  for i in $(eval echo "{1..$prop_count}"); do
    read prop
    props+=("$prop")
  done
  read opt_count
  opts=()
  for i in $(eval echo "{1..$opt_count}"); do
    read opt
    opts+=("$opt")
  done
  
  echo "$model $properties"
  echo "$prop_count $opt_count"
  for prop in "${props[@]}"; do
    echo "$prop"
    for opt in "${opts[@]}"; do
      echo "-const $opt"
      bench_one $times ../examples/$model ../examples/$properties "$prop" "-const $opt"
    done 
  done
done

