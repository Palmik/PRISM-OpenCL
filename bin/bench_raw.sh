#!/bin/bash

get_state_count() {
  sed -n 's:.*States\: *\([0-9]*\).*:\1:p' $1
}

get_transition_count() {
  sed -n 's:.*Transitions\: *\([0-9]*\).*:\1:p' $1
}

get_model_checking_time() {
  sed -n 's:.*Time for model checking\: \([0-9,\.]*\) seconds.*:\1:p' $1
}

get_model_construction_time() {
  sed -n 's:.*Time for model construction\: \([0-9,\.]*\) seconds.*:\1:p' $1
}

bench_one() {
  times=$1
  shift

  for i in $(eval echo "{1..$times}"); do
    ./prism.bat -s $* &> prism_bench.out
   
    if [[ $i -eq 1 ]]; then
      echo $(get_state_count prism_bench.out)
      echo $(get_transition_count prism_bench.out)
    fi 

    echo $(get_model_checking_time prism_bench.out)
  done

  for i in $(eval echo "{1..$times}"); do
    ./prism.bat -s -opencl $* &> prism_bench.out
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
    props+=($prop)
  done
  read opt_count
  opts=()
  for i in $(eval echo "{1..$opt_count}"); do
    read opt
    opts+=($opt)
  done
  
  echo "$model $properties"
  echo "$prop_count $opt_count"
  for prop in "${props[@]}"; do
    echo "-prop $prop"
    for opt in "${opts[@]}"; do
      echo "-const $opt"
      args="../examples/$model ../examples/$properties -prop $prop -const $opt"
      bench_one $times $args 
    done 
  done
done

