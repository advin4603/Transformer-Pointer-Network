from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput
from pointer import Pointer
import torch
import inspect
from torch.nn import CrossEntropyLoss
from typing import *

class T5Pointer(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.pointer = Pointer(config.d_model, config.num_heads * config.num_layers)
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "encoder_input_ids": kwargs["encoder_input_ids"]
        }
    
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        if (
            self.config.is_encoder_decoder
            and hasattr(self, "encoder")
            and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}
        model_kwargs["encoder_input_ids"] = model_kwargs["input_ids"]
        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed."
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 3. In the presence of `inputs_embeds` for text models:
        # - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
        # doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
        # input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
        # - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
        # pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(
                    inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
                )
                if not has_inputs_embeds_forwarding:
                    raise ValueError(
                        f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                        "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                        "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                    )
                # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
                # the attention mask) can rely on the actual model input.
                model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                    inputs, bos_token_id, model_kwargs=model_kwargs
                )
            else:
                if inputs is not None:
                    raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs
    
    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = True
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        encoder_input_ids = None
        
        if "encoder_input_ids" in kwargs:
            encoder_input_ids = kwargs.pop("encoder_input_ids")
        base_output: Seq2SeqLMOutput = super().forward(*args, **kwargs)
        attention_scores = torch.cat(base_output.cross_attentions, dim=-3).transpose(-3, -2).transpose(-2, -1)
        hidden_states = base_output.decoder_hidden_states[-1]
        input_ids = kwargs.get("input_ids") if len(args) == 0 else args[0]
        
        if input_ids is None:
            input_ids = encoder_input_ids
        base_output.logits, pgen = self.pointer(input_ids, attention_scores, hidden_states, base_output.logits)
        
        if kwargs.get("labels") is not None or len(args) >= 12:
            labels = kwargs.get("labels", None)
            if labels is None:
                labels = args[11]
            
            # Add pgen penalty term
            loss_pgn = (pgen - 0.5) ** 2 / 0.25
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(base_output.logits.device)
            base_output.loss = loss_fct(base_output.logits.view(-1, base_output.logits.size(-1)), labels.view(-1)) + loss_pgn.mean()
        
        return base_output
    
if __name__ == "__main__":
    a = T5Pointer.from_pretrained("t5-small")
    a(input_ids=torch.ones(10, 5, dtype=torch.long), labels=torch.ones(10, 7, dtype=torch.long))
