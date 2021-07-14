import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from evaluation.tokenizer import get_seq, untokenize, tokenize_string_with_spans
from evaluation.preprocessor import preprocess_dataset, SpanDataset, separate_spans

def predict_sentence(sentence, tags, tokenizer, model):
    test_sentence = "de man vraagt de vrouw te vertrekken"
    test_tags = [1, 1, 0, 2, 2, 0, 0]
    input_ids, input_tags, input_masks = tokenize_string_with_spans(tokenizer=tokenizer, words=sentence.split(), tags=tags)
    input_subj_tags, input_obj_tags = separate_spans([input_tags])
    input_subj_tags, input_obj_tags = input_subj_tags[0], input_obj_tags[0]

    input_ids, input_masks, input_subj_tags, input_obj_tags = torch.tensor(input_ids), torch.tensor(input_masks), torch.tensor(input_subj_tags), torch.tensor(input_obj_tags)
    input_ids, input_masks, input_subj_tags, input_obj_tags = input_ids.view(1, -1), input_masks.view(1, -1), input_subj_tags.view(1, -1), input_obj_tags.view(1, -1)
    prediction = model.forward(input_ids, input_masks, input_subj_tags, input_obj_tags)
    return prediction.argmax(dim=1).item()

def analyse_test_results(test_trues, test_preds):
    trues = [t.item() for t in test_trues]
    preds = [p.argmax().item() for p in test_preds]
    return confusion_matrix(trues, preds)


def detokenize(id_seq, tokenizer):
    return untokenize(tokenizer, get_seq(id_seq))


def load_orig_test():
    orig_dataset = preprocess_dataset(name='orig', already_prepped=False, sentence_cutoff=60, max_len=140)
    return [(list(get_seq(torch.tensor(p[0]))), p[4]) for p in SpanDataset(orig_dataset[2])]


def split_orig_swap(test_inputs, test_preds, test_trues):
    orig_test = load_orig_test()
    orig_inputs = [i[0] for i in orig_test]
    orig_cases = [(i, p, t) for i, p, t in zip(test_inputs, test_preds, test_trues) if list(get_seq(i)) in orig_inputs]
    swap_cases = [(i, p, t) for i, p, t in zip(test_inputs, test_preds, test_trues) if list(get_seq(i)) not in orig_inputs]
    return orig_cases, swap_cases


def analyse_test_cases(test_inputs, test_preds, test_trues):
    orig_cases, swap_cases = split_orig_swap(test_inputs, test_preds, test_trues)
    orig_inputs, orig_preds, orig_trues = zip(*orig_cases)
    swap_inputs, swap_preds, swap_trues = zip(*swap_cases)
    orig_metrics = analyse_test_results(orig_trues, orig_preds)
    swap_metrics = analyse_test_results(swap_trues, swap_preds)
    return orig_metrics, swap_metrics


def analyse_test(trainer):
    # first_results = trainer.train_epoch(device='cpu', epoch_i=1)
    test_loss, test_acc, test_inputs, test_preds, test_trues = trainer.predict_epoch(eval_set='test', device='cpu', epoch_i=1)
    total_metrics = analyse_test_results(test_trues, test_preds)
    orig_metrics, swap_metrics = analyse_test_cases(test_inputs, test_preds, test_trues)
    return total_metrics, orig_metrics, swap_metrics