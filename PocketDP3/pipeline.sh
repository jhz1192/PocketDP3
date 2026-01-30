#!/bin/bash

gpu_id=${1}

tasks=(
  beat_block_hammer
  click_alarmclock
  click_bell
  hanging_mug
  move_can_pot
  move_pillbottle_pad
  pick_diverse_bottles
  pick_dual_bottles
  place_bread_basket
  place_bread_skillet
  place_burger_fries
  place_cans_plasticbox
  place_empty_cup
  place_object_stand
  place_phone_stand
  scan_object
  stack_bowls_three
  stamp_seal
  turn_switch
)

for task in "${tasks[@]}"; do
  echo "Current task: $task"
  bash train.sh $task demo_clean 50 0 $gpu_id
  bash eval.sh $task demo_clean demo_clean 50 0 $gpu_id
done