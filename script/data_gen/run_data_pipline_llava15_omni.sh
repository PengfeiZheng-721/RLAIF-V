# An example of LLaVA 1.5 7B generate - divide and conquer - OmniLMM 12B feedback

## step1: diverse gen
model_path=liuhaotian/llava-v1.5-7b
ques_dir=./examples # directory of the input file
ques_file=detail_test_30_input # name of the input file, without '.jsonl'
ans_dir=./results # directory of the answer files
start=0
end=-1
bash ./script/data_gen/llava15/llava15_diverse_gen.sh \
$model_path \
$ans_dir \
$ques_dir \
${ques_file}.jsonl \
$start \
$end \
8 # num of gpus that can be used


# step2: spaCy divide
ans_file=${ans_dir}/diverse_gen_llava15_${start}-${end}_${ques_file} # path of the generation result file
bash ./script/data_gen/divide_and_conquer/spacy_divide.sh \
${ans_file}.jsonl $start $end

## step3: LLaVA-NeXT-34B autocheck
model_path=openbmb/NeXT-34B
check_ques_file=diverse_gen_llava15_${start}-${end}_${ques_file}.spacy_divide.jsonl
bash ./script/data_gen/next34b/next34b_autocheck.sh \
  $model_path $ans_dir $check_ques_file 0 -1 8

# step4: construct preference pairs
gq_file=diverse_gen_llava15_${start}-${end}_${ques_file}.spacy_divide.jsonl # name of the divided answer file
feedback_file=autocheck_next34b_0--1_diverse_gen_llava15_${start}-${end}_${ques_file}.spacy_divide.jsonl # name of the feedback result file
bash ./script/data_gen/construct_pairs.sh \
  ${ans_dir}/${feedback_file} \
  ${ans_dir}/${gq_file} \
  2 # num of the sample pairs for each instruction

# balance win - lose answer length
pairs_file=autocheck_next34b_0--1_diverse_gen_llava15_${start}-${end}_${ques_file}.spacy_divide.jsonl_pair_diff1_samp2.jsonl
balanced_pairs_file=autocheck_next34b_0--1_diverse_gen_llava15_${start}-${end}_${ques_file}.spacy_divide.jsonl_pair_diff1_samp2_balanceshort.jsonl
python ./utils/get_pairs_filter_shorten.py \
  --path ${ans_dir}/$pairs_file \
  --save_path ${ans_dir}/$balanced_pairs_file
