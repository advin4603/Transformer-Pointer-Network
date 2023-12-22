from torch import nn
import torch

class Pointer(nn.Module):
    def __init__(self, hidden_state_dimensions: int, attention_dimensions: int):
        super().__init__()
        self.hidden_state_dimensions = hidden_state_dimensions
        self.attention_dimensions = attention_dimensions
        
        self.p_gen_probability_layer = nn.Sequential(
            # nn.Linear(hidden_state_dimensions, hidden_state_dimensions // 2),
            # nn.Linear(hidden_state_dimensions // 2, hidden_state_dimensions),
            nn.Linear(hidden_state_dimensions, 1),
            nn.Sigmoid()
        )
        
        self.input_vocabulary_addition_layer = nn.Sequential(
            # nn.Linear(attention_dimensions, attention_dimensions // 2),
            # nn.Linear(attention_dimensions // 2, attention_dimensions),
            nn.Linear(attention_dimensions, 1),
            nn.ReLU()
        )
        
        self.pgen_list = []
    
    def forward(self, input_ids, attentions, hidden_states, output_vocabulary_probabilities, attention_mask = None):
        """
        input_ids: [Batch_size, enc_seq_len] or [enc_seq_len]
        attentions: [Batch_size, dec_seq_length ,enc_seq_length , num_heads*num_layers] or [dec_seq_length, enc_seq_length, num_heads * num_layers]
        hidden_states: [Batch_size, dec_seq_length ,hidden_state_dimensions] or [dec_seq_length ,hidden_state_dimensions]
        output_vocabulary_probabilities: [Batch_size, dec_seq_length, vocabulary_size] or [vocabulary_size]

        Returns: [Batch_size, vocabulary_size] or [vocabulary_size]
        """
        input_ids = input_ids.unsqueeze(-2).expand(-1, hidden_states.shape[-2], -1)
        p_gen = self.p_gen_probability_layer(hidden_states)
        
        self.pgen_list.append(p_gen.item())
        
        input_vocabulary_addition = self.input_vocabulary_addition_layer(attentions).squeeze()
        return (p_gen * output_vocabulary_probabilities).scatter_add(-1, input_ids, (1 - p_gen)* input_vocabulary_addition), p_gen
