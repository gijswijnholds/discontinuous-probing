import torch

VERB_IDX = 99

train_configs = [{'name': 'diffsep_anw4', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': 'diffsep_anw5', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1}]


train_configs_____ = [{'name': 'diffsep_anw1', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.NLLLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': 'diffsep_anw2', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.NLLLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': 'diffsep_anw3', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.NLLLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1}]

train_configs____ = [{'style': 'SynthAttention', 'name': 'diffsep_anw1', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'style': 'SynthAttention', 'name': 'diffsep_anw2', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'style': 'SynthAttention', 'name': 'diffsep_anw3', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1}
    ]
train_configs___ = [{'style': 'SynthAttention', 'name': 'diffsep', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'style': 'SynthAttention', 'name': 'diffsep_split', 'optim': torch.optim.AdamW, 'lr': 0.00003,
                  'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1}
                 ,{'style': 'SynthAttention', 'name': 'diffsep_split2', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'style': 'SynthAttention', 'name': 'diffsep_split3', 'optim': torch.optim.AdamW, 'lr': 0.00003,
                  'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1}
                 ]

train_configs__ = [{'style': 'Attention', 'name': None, 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'style': 'Attention', 'name': None, 'optim': torch.optim.AdamW, 'lr': 0.0003,
                  'loss_fn': torch.nn.CrossEntropyLoss, 'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 100, 'run': 1},
                 {'style': 'SynthAttention', 'name': None, 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'style': 'SynthAttention', 'name': 'diff', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'style': 'SynthAttention', 'name': 'diffsep', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'style': 'SynthAttention', 'name': 'diffsep_anw', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1}
                 ]

train_configs_ = [{'style': 'Attention', 'name': None, 'optim': torch.optim.AdamW, 'lr': 0.0003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 1, 'batch_size': 20, 'run': 1},
                 {'style': 'Attention', 'name': None, 'optim': torch.optim.AdamW, 'lr': 0.000003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 1, 'run': 1},
                 {'style': 'Attention', 'name': None, 'optim': torch.optim.AdamW, 'lr': 0.000003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': False, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 1, 'run': 1},
                 {'style': 'Attention', 'name': 'orig', 'optim': torch.optim.AdamW, 'lr': 0.0003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': False, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 20, 'run': 1}]

train_configs_full_span_finetune = [{'style': 'AttentionFullSpan', 'name': None, 'optim': torch.optim.AdamW, 'lr': 0.0003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': False, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 20, 'run': 1},
                 {'style': 'AttentionFullSpan', 'name': 'orig', 'optim': torch.optim.AdamW, 'lr': 0.0003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': False, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 20, 'run': 1}]

# train_configs_full_span_frozen = [{'style': 'AttentionFullSpan', 'name': None, 'optim': torch.optim.AdamW, 'lr': 0.0003, 'loss_fn': torch.nn.CrossEntropyLoss,
#                   'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 20, 'run': 1},
#                  {'style': 'AttentionFullSpan', 'name': 'orig', 'optim': torch.optim.AdamW, 'lr': 0.0003, 'loss_fn': torch.nn.CrossEntropyLoss,
#                   'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 20, 'run': 1}]

train_configs_att_b20 = [{'style': 'Attention', 'name': None, 'optim': torch.optim.AdamW, 'lr': 0.0003, 'loss_fn': torch.nn.CrossEntropyLoss,
                          'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 20, 'run': 1},
                         {'style': 'Attention', 'name': 'orig', 'optim': torch.optim.AdamW, 'lr': 0.0003, 'loss_fn': torch.nn.CrossEntropyLoss,
                          'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 20, 'run': 1}]


train_configs_attention_bertje = [{'style': 'Attention', 'name': None, 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'style': 'Attention', 'name': 'orig', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1}]

train_configs_mbert = [{'name': None, 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'bert-base-multilingual-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': None, 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': False, 'bert_model': 'bert-base-multilingual-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': 'orig', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'bert-base-multilingual-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': 'orig', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': False, 'bert_model': 'bert-base-multilingual-cased', 'epochs': 7, 'batch_size': 10, 'run': 1}
                 ]


train_configs_robbert = [{'name': None, 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'pdelobelle/robbert-v2-dutch-base', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': None, 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': False, 'bert_model': 'pdelobelle/robbert-v2-dutch-base', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': 'orig', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'pdelobelle/robbert-v2-dutch-base', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': 'orig', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': False, 'bert_model': 'pdelobelle/robbert-v2-dutch-base', 'epochs': 7, 'batch_size': 10, 'run': 1}
                 ]

train_configs_bertje = [{'name': None, 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': None, 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': False, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': 'orig', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': True, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1},
                 {'name': 'orig', 'optim': torch.optim.AdamW, 'lr': 0.00003, 'loss_fn': torch.nn.CrossEntropyLoss,
                  'freeze': False, 'bert_model': 'GroNLP/bert-base-dutch-cased', 'epochs': 7, 'batch_size': 10, 'run': 1}
                 ]