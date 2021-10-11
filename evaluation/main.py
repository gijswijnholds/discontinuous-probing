from .sparse import SparseVA
from .preprocessing import prepare_datasets
from .trainer import Trainer

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


def setup_trainer(grammar_file: str, bert_name: str, freeze: bool):
    model = SparseVA(model_name=bert_name, freeze=freeze, dim=768, selection_h=128)
    train_ds, val_ds, test_ds = prepare_datasets(grammar_file)
    return Trainer(model=model,
                   train_dataset=train_ds,
                   val_dataset=val_ds,
                   test_dataset=test_ds,
                   batch_size_train=32, batch_size_val=128, batch_size_test=128,
                   word_pad=3,
                   optim_constructor=AdamW,
                   lr=1e-05,
                   loss_fn=CrossEntropyLoss())
