I had some issues with prediction, that's why there are several output files.
My model using GenerationWrapper/_generate function in some cases predicts EOS as a first token
and then stops (see multiwoz_outputs_first_sent.txt). In order to overcome this
behaviour I forced token by token prediction using GenerationWrapper/_generate_step function,
where in case the first EOS met is ignored and prediction ends only when the next EOS is
predicted (see multiwoz_outputs_first_sent_forced.txt and multiwoz_outputs_full_forced.txt).

With this second approach I'm getting warning:

"We strongly recommend passing in an `attention_mask` since your input_ids may be padded.
See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.
You may ignore this warning if your `pad_token_id` (50256) is identical to the `bos_token_id` (50256),
`eos_token_id` (50256), or the `sep_token_id` (None), and your input is not padded."

and the results are repetitive sometimes, eg:

23 context> I'd like a train that is departing from Cambridge and is going to London Liverpool street.
23 answer>  What day and time would you like to travel?What day and time would you like to travel?

I still believe "ignore the first EOS" approach works better, but I'd appreciate suggestions on
how this could be done better :)

